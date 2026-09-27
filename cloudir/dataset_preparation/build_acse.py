from __future__ import annotations

import importlib
import json
import random
import shutil
from typing import Any, Generator

from cloudir.ai_models.coach_model import generate_initial_turn_from_dataset
from cloudir.ai_models.coach_model import unload_coach_model
from cloudir.ai_models.image_model import unload_image_model
from cloudir.ai_models.security_model import unload_security_model, write_incident_timeline
from cloudir.paths import GENERATED_EVIDENCE_DIR, WORKSPACE_ROOT as PROJECT_ROOT
from cloudir.scenario.evidence_template_strategy import apply_evidence_template_strategy
from cloudir.scenario.incident_timeline import (
    TIMELINE_FILE,
    action_anchor,
    apply_timeline_evidence,
    build_turn_evidence,
    finalise_incident_timeline,
    enforce_action_anchor,
    remove_decoy_mentions,
    marking_scheme,
    save_incident_timeline,
    threat_model_services,
    timeline_prompt_block,
)
from cloudir.scenario.turn_validation import validate_evidence_facts

from .build_runtime import SCENARIO_DATA_DIR, build_runtime_files
from .normalise_threat import normalise_threat
from .parse_architecture import parse_architecture
from .select_case import processed_dir, select_acse_case, scenario_slug


TURN_1_DIR = SCENARIO_DATA_DIR / "turns" / "turn_1"
TURN_1_EVIDENCE_DIR = GENERATED_EVIDENCE_DIR / "turn_1"
TURN_1_MAX_ATTEMPTS = 3
TIMELINE_MAX_ATTEMPTS = 3


def build_acse_dataset_scenario(
    scenario_id: str = "identity-management",
) -> dict[str, Any]:
    """
    Runs the full ACSE dataset preparation pipeline without streaming.

    This is useful for terminal testing.
    """

    final_result: dict[str, Any] | None = None

    for event in stream_build_acse_dataset_scenario(scenario_id):
        if event["status"] == "error":
            raise RuntimeError(event["message"])

        if event["status"] == "complete":
            final_result = event["result"]

    if final_result is None:
        raise RuntimeError("ACSE dataset preparation did not complete.")

    return final_result


def stream_build_acse_dataset_scenario(
    scenario_id: str = "identity-management",
) -> Generator[dict[str, Any], None, None]:
    """
    Streams ACSE preparation events for the Flask preparation page.

    The Flask route can convert each yielded dictionary into server-sent events.
    """

    try:
        slug = scenario_slug(scenario_id)
        scenario_processed_dir = processed_dir(scenario_id)

        yield _event(
            stage="select_case",
            label="Select ACSE case",
            status="running",
            message=f"Selecting the {scenario_id} row from acse_eval.jsonl.",
            kind="code",
        )

        selected_case = select_acse_case(scenario_id)

        yield _event(
            stage="select_case",
            label="Select ACSE case",
            status="success",
            message="Created selected_acse_case.json.",
            kind="code",
            output=str((scenario_processed_dir / "selected_acse_case.json").relative_to(PROJECT_ROOT)),
        )

        yield _event(
            stage="parse_architecture",
            label="Architecture VLM parsing",
            status="running",
            message="Sending architecture.png to the local VLM.",
            kind="ai",
        )

        architecture_facts = parse_architecture(scenario_id)
        unload_image_model()

        yield _event(
            stage="parse_architecture",
            label="Architecture VLM parsing",
            status="success",
            message="Created architecture_facts.json.",
            kind="ai",
            output=str((scenario_processed_dir / "architecture_facts.json").relative_to(PROJECT_ROOT)),
        )

        yield _event(
            stage="normalise_threat",
            label="Threat model normalisation",
            status="running",
            message=(
                "Sending threat-model.json and architecture_facts.json to the "
                "security text model."
            ),
            kind="ai",
        )

        normalised_threat_model = normalise_threat(scenario_id)
        unload_security_model()

        yield _event(
            stage="normalise_threat",
            label="Threat model normalisation",
            status="success",
            message="Created normalised_threat_model.json.",
            kind="ai",
            output=str((scenario_processed_dir / "normalised_threat_model.json").relative_to(PROJECT_ROOT)),
        )

        yield _event(
            stage="build_runtime",
            label="Runtime scenario files",
            status="running",
            message=(
                "Building scenario_seed.json, hidden_truth.json, and "
                "scenario_config.json."
            ),
            kind="code",
        )

        # Clear the old scenario's turns before its runtime files are replaced, so a
        # build that fails later never pairs this scenario's config with another's turns.
        _reset_turn_1_outputs()
        runtime_files = build_runtime_files(scenario_id)

        yield _event(
            stage="build_runtime",
            label="Runtime scenario files",
            status="success",
            message="Created scenario_seed.json, hidden_truth.json, and scenario_config.json.",
            kind="code",
            output="data/runtime/",
        )

        scenario_config = runtime_files["scenario_config"]

        yield _event(
            stage="incident_timeline",
            label="Incident timeline",
            status="running",
            message="Security AI writes one incident timeline; every turn's evidence is built from it.",
            kind="ai",
        )

        # A timeline from an earlier build must never pair with this scenario's turns.
        (SCENARIO_DATA_DIR / TIMELINE_FILE).unlink(missing_ok=True)
        retry_instruction, draft = "", None

        try:
            for attempt in range(1, TIMELINE_MAX_ATTEMPTS + 1):
                # A draft that was never parsed is not shown back to the model.
                rejected, draft = draft, None
                try:
                    draft = write_incident_timeline(
                        normalised_threat_model=normalised_threat_model,
                        architecture_facts=architecture_facts,
                        turn_goals=scenario_config.get("scenario_progression", {}).get("incident_response_stages", []),
                        retry_instruction=retry_instruction,
                        scenario_services=threat_model_services(normalised_threat_model),
                        attempt=attempt,
                        rejected_timeline=rejected,
                        scenario_id=scenario_id,
                    )
                    timeline = finalise_incident_timeline(draft, scenario_config, threat_model=normalised_threat_model)
                    break
                except ValueError as exc:
                    if attempt == TIMELINE_MAX_ATTEMPTS:
                        raise

                    retry_instruction = str(exc)[:900]

                    yield _event(
                        stage="incident_timeline",
                        label="Incident timeline",
                        status="running",
                        message=(
                            f"Timeline was rejected: {retry_instruction[:300]} "
                            f"Regenerating (attempt {attempt + 1} of {TIMELINE_MAX_ATTEMPTS})."
                        ),
                        kind="ai",
                    )
        finally:
            unload_security_model()

        save_incident_timeline(SCENARIO_DATA_DIR, timeline)

        yield _event(
            stage="incident_timeline",
            label="Incident timeline",
            status="success",
            message=f"Created incident_timeline.json with {len(timeline['events'])} events.",
            kind="ai",
            output=f"data/runtime/{TIMELINE_FILE}",
        )

        yield _event(
            stage="coach_turn_1",
            label="Coach Turn 1 generation",
            status="running",
            message="Sending prepared dataset scenario to the Coach AI.",
            kind="ai",
        )

        retry_instruction = ""
        turn_1_evidence = build_turn_evidence(timeline, scenario_config, 1)
        anchor = action_anchor(timeline, turn_1_evidence, 1)

        try:
            for attempt in range(1, TURN_1_MAX_ATTEMPTS + 1):
                try:
                    turn_1 = generate_initial_turn_from_dataset(
                        scenario_seed=runtime_files["scenario_seed"],
                        hidden_truth=runtime_files["hidden_truth"],
                        scenario_config=scenario_config,
                        retry_instruction=retry_instruction,
                        evidence_brief=timeline_prompt_block(timeline, turn_1_evidence, anchor),
                    )
                    # The automated-response drift repair swaps in a hardcoded turn about a
                    # different incident; with timeline evidence it would contradict it.
                    apply_evidence_template_strategy(
                        generated_turn=turn_1,
                        scenario_config=scenario_config,
                        turn_number=1,
                    )
                    # Evidence facts always come from the timeline; the coach wrote the text.
                    apply_timeline_evidence(turn_1, turn_1_evidence)
                    # The learner context must not introduce the weak item's principal.
                    remove_decoy_mentions(turn_1, timeline, turn_1_evidence)
                    # The best action must name what only the strong item shows.
                    enforce_action_anchor(turn_1.get("actions"), anchor)
                    # The security judge's marking scheme comes from the timeline, not the coach.
                    turn_1["expected_outcomes"] = marking_scheme(timeline, turn_1_evidence, anchor)
                    # Validates evidence content against support roles before writing.
                    _save_turn_1_files(turn_1)
                    break
                except ValueError as exc:
                    if attempt == TURN_1_MAX_ATTEMPTS:
                        raise

                    retry_instruction = str(exc)[:900]

                    yield _event(
                        stage="coach_turn_1",
                        label="Coach Turn 1 generation",
                        status="running",
                        message=(
                            f"Turn 1 was rejected: {retry_instruction[:300]} "
                            f"Regenerating (attempt {attempt + 1} of {TURN_1_MAX_ATTEMPTS})."
                        ),
                        kind="ai",
                    )
        finally:
            unload_coach_model()

        yield _event(
            stage="coach_turn_1",
            label="Coach Turn 1 generation",
            status="success",
            message=(
                "Created turn_config.json, actions.json, evidence_facts.json, "
                "and expected_outcomes.json."
            ),
            kind="ai",
            output="data/runtime/turns/turn_1/",
        )

        yield _event(
            stage="render_evidence",
            label="Evidence screenshot rendering",
            status="running",
            message="Rendering Turn 1 AWS-style evidence screenshots.",
            kind="code",
        )

        render_result = _render_turn_1_evidence()

        yield _event(
            stage="render_evidence",
            label="Evidence screenshot rendering",
            status="success",
            message="Rendered Turn 1 evidence screenshots.",
            kind="code",
            output="output/generated_evidence/turn_1/",
            data=render_result,
        )

        validation = _validate_outputs(scenario_id)

        yield _event(
            stage="complete",
            label="Dataset scenario ready",
            status="complete",
            message="ACSE dataset scenario is ready. CloudIR runtime can now start.",
            kind="output",
            result={
                "scenario_id": scenario_id,
                "selected_case": selected_case,
                "architecture_facts": architecture_facts,
                "normalised_threat_model": normalised_threat_model,
                "runtime_files": {
                    "scenario_seed": "data/runtime/scenario_seed.json",
                    "hidden_truth": "data/runtime/hidden_truth.json",
                    "scenario_config": "data/runtime/scenario_config.json",
                    "turn_1": "data/runtime/turns/turn_1/",
                    "evidence": "output/generated_evidence/turn_1/",
                },
                "validation": validation,
            },
        )

    except Exception as exc:
        yield _event(
            stage="error",
            label="Dataset preparation failed",
            status="error",
            message=str(exc),
            kind="error",
        )


def _reset_turn_1_outputs() -> None:
    # A newly prepared scenario must not inherit later turns from a previous run.
    # Existing attempts are archived by the application run manager before build.
    for directory in [SCENARIO_DATA_DIR / "turns", GENERATED_EVIDENCE_DIR]:
        if directory.exists():
            shutil.rmtree(directory)

    TURN_1_DIR.mkdir(parents=True, exist_ok=True)
    TURN_1_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _save_turn_1_files(turn_1: dict[str, Any]) -> None:
    required_keys = {
        "turn_config",
        "actions",
        "evidence_facts",
        "expected_outcomes",
    }

    missing = required_keys - set(turn_1.keys())

    if missing:
        raise ValueError(f"Coach Turn 1 output is missing keys: {sorted(missing)}")

    validate_evidence_facts(turn_1["evidence_facts"])
    shuffle_turn_options(turn_1, turn_number=1)

    _write_json(TURN_1_DIR / "turn_config.json", turn_1["turn_config"])
    _write_json(TURN_1_DIR / "actions.json", turn_1["actions"])
    _write_json(TURN_1_DIR / "evidence_facts.json", turn_1["evidence_facts"])
    _write_json(TURN_1_DIR / "expected_outcomes.json", turn_1["expected_outcomes"])


def shuffle_turn_options(turn: dict[str, Any], turn_number: int) -> None:
    for key in ("actions", "evidence_facts"):
        items = turn.get(key)

        if not isinstance(items, list) or len(items) < 2:
            continue

        rng = random.SystemRandom()
        rng.shuffle(items)
        move_best_item_away_from_first(items, key)


def move_best_item_away_from_first(items: list[Any], key: str) -> None:
    if len(items) < 2 or not isinstance(items[0], dict):
        return

    role_key = "choice_role" if key == "actions" else "support_role"
    best_value = "best" if key == "actions" else "strong"

    if str(items[0].get(role_key, "")).strip().lower() != best_value:
        return

    best_item = items.pop(0)
    insert_at = random.SystemRandom().randint(1, len(items))
    items.insert(insert_at, best_item)


def _automated_security_response_evidence() -> list[dict[str, Any]]:
    return [
        {
            "id": "security_finding_trigger",
            "title": "Security Finding Trigger",
            "type": "guardduty",
            "summary": (
                "Security monitoring shows the finding that triggered automated remediation."
            ),
            "why_it_may_matter": (
                "This directly identifies the finding type, affected workflow resource, severity, and trigger context."
            ),
            "support_role": "strong",
            "template": "guardduty",
            "facts": {
                "finding_id": "custom-finding-7421",
                "finding_type": "SecurityHub.CustomFinding/RemediationInput",
                "severity": "High",
                "resource": "securityhub-remediation-workflow",
                "principal": "events.amazonaws.com/securityhub-remediation-rule",
                "remote_ip": "AWS Internal",
                "region": "ap-southeast-1",
                "first_seen": "2023-10-01T10:19:12Z",
                "last_seen": "2023-10-01T10:19:12Z",
                "summary": (
                    "A high-severity finding references the remediation workflow target and needs validation before automation proceeds."
                ),
            },
        },
        {
            "id": "eventbridge_start_execution",
            "title": "EventBridge StartExecution Event",
            "type": "cloudtrail",
            "summary": (
                "CloudTrail shows EventBridge starting the remediation state machine."
            ),
            "why_it_may_matter": (
                "This helps connect the finding to the workflow, but it needs the finding content to prove why automation started."
            ),
            "support_role": "partial",
            "template": "cloudtrail",
            "facts": {
                "event_name": "StartExecution",
                "event_source": "states.amazonaws.com",
                "user": "events.amazonaws.com/securityhub-remediation-rule",
                "source_ip": "AWS Internal",
                "mfa": "service",
                "event_time": "2023-10-01T10:21:35Z",
                "region": "ap-southeast-1",
                "error_code": "-",
                "risk_signal": "EventBridge started remediation for finding custom-finding-7421.",
                "event_id": "evt-securityhub-remediation-001",
                "user_agent": "events.amazonaws.com",
                "recipient_account_id": "123456789012",
                "request_parameters": "stateMachineArn=securityhub-remediation-workflow; findingId=custom-finding-7421",
                "related_events": [
                    ["10:19:12 UTC", "BatchImportFindings", "securityhub.amazonaws.com", "AWS Internal", "Success"],
                    ["10:20:04 UTC", "PutEvents", "events.amazonaws.com", "AWS Internal", "Success"],
                    ["10:21:35 UTC", "StartExecution", "events.amazonaws.com/securityhub-remediation-rule", "AWS Internal", "Success"],
                ],
            },
        },
        {
            "id": "remediation_role_access_key",
            "title": "Remediation Role Credential Review",
            "type": "access_key",
            "summary": (
                "Credential review shows temporary activity associated with the remediation role."
            ),
            "why_it_may_matter": (
                "This is useful context, but role permissions alone do not prove the finding was malicious."
            ),
            "support_role": "weak",
            "template": "access_key",
            "facts": {
                "access_key_id": "AKIAREM7421SAFETY999",
                "owner": "RemediationRole/session/securityhub-remediation",
                "status": "Active",
                "last_used_service": "ssm.amazonaws.com",
                "last_used_region": "ap-southeast-1",
                "last_used_time": "2023-10-01T10:22:04Z",
                "source_ip": "AWS Internal",
            },
        },
    ]


def _render_turn_1_evidence() -> dict[str, Any]:
    """
    Calls the existing evidence renderer without putting rendering logic here.

    This keeps ACSE preparation as orchestration only.

    The existing CloudIR runtime calls generate_evidence_for_turn with:
    - turn_number
    - root_dir

    So this preparation stage should call it the same way.
    """

    module = importlib.import_module("cloudir.evidence.generate_for_turn")

    renderer = getattr(module, "generate_evidence_for_turn", None)

    if not callable(renderer):
        raise AttributeError(
            "Could not find generate_evidence_for_turn() in "
            "cloudir/evidence/generate_for_turn.py."
        )

    result = renderer(
        turn_number=1,
        root_dir=PROJECT_ROOT,
    )

    return {
        "renderer": "generate_evidence_for_turn",
        "kwargs": {
            "turn_number": 1,
            "root_dir": str(PROJECT_ROOT),
        },
        "result": result,
    }
def _validate_outputs(scenario_id: str) -> dict[str, Any]:
    scenario_processed_dir = processed_dir(scenario_id)

    required_files = [
        scenario_processed_dir / "selected_acse_case.json",
        scenario_processed_dir / "architecture_facts.json",
        scenario_processed_dir / "normalised_threat_model.json",
        SCENARIO_DATA_DIR / "scenario_seed.json",
        SCENARIO_DATA_DIR / "hidden_truth.json",
        SCENARIO_DATA_DIR / "scenario_config.json",
        TURN_1_DIR / "turn_config.json",
        TURN_1_DIR / "actions.json",
        TURN_1_DIR / "evidence_facts.json",
        TURN_1_DIR / "expected_outcomes.json",
    ]

    missing_files = [str(path.relative_to(PROJECT_ROOT)) for path in required_files if not path.exists()]
    screenshots = sorted(TURN_1_EVIDENCE_DIR.glob("*.png"))

    if missing_files:
        raise FileNotFoundError(
            "Dataset preparation completed but required files are missing: "
            f"{missing_files}"
        )

    if not screenshots:
        raise FileNotFoundError(
            "Dataset preparation completed but no Turn 1 evidence screenshots "
            "were found in output/generated_evidence/turn_1/."
        )

    return {
        "required_files_present": True,
        "screenshot_count": len(screenshots),
        "screenshots": [
            str(path.relative_to(PROJECT_ROOT)) for path in screenshots
        ],
    }


def _event(
    stage: str,
    label: str,
    status: str,
    message: str,
    kind: str,
    output: str | None = None,
    data: Any | None = None,
    result: Any | None = None,
) -> dict[str, Any]:
    event = {
        "stage": stage,
        "label": label,
        "status": status,
        "kind": kind,
        "message": message,
    }

    if output is not None:
        event["output"] = output

    if data is not None:
        event["data"] = data

    if result is not None:
        event["result"] = result

    return event


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


prepare_acse_dataset_scenario = build_acse_dataset_scenario
stream_prepare_acse_dataset_scenario = stream_build_acse_dataset_scenario
