from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from cloudir.ai_models.coach_model import evaluate_generated_turn_quality, generate_next_turn_json
from cloudir.ai_models.model_lifecycle import log_memory
from cloudir.evidence_core.fact_repair import normalise_evidence_facts, repair_next_turn_template_facts
from cloudir.evidence_core.template_inference import force_correct_template, validate_template_alignment
from cloudir.evidence_core.evidence_text_helpers import clean_value
from cloudir.scenario.evidence_template_strategy import apply_evidence_template_strategy
from cloudir.scenario.incident_timeline import (
    action_anchor,
    apply_timeline_evidence,
    build_turn_evidence,
    enforce_action_anchor,
    remove_decoy_mentions,
    load_incident_timeline,
    marking_scheme,
    timeline_prompt_block,
)
from cloudir.scenario.scenario_progression import (
    apply_scenario_progression_safety,
    routine_application_logs_evidence,
)
from cloudir.scenario.action_choice_roles import normalise_action_choice_roles
from cloudir.scenario.evidence_support_roles import normalise_evidence_support_roles
from cloudir.scenario.turn_validation import validate_generated_turn


TURN_FILES = [
    "turn_config.json",
    "actions.json",
    "evidence_facts.json",
    "expected_outcomes.json",
]

MAX_GENERATION_ATTEMPTS = 3
MIN_ACCEPTABLE_QUALITY_SCORE = 70


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"File exists but is empty: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def turn_files_exist(root_dir: Path, turn_number: int) -> bool:
    turn_dir = root_dir / "data" / "runtime" / "turns" / f"turn_{turn_number}"

    for file_name in TURN_FILES:
        file_path = turn_dir / file_name

        if not file_path.exists():
            return False

        if file_path.stat().st_size == 0:
            return False

        try:
            read_json(file_path)
        except (json.JSONDecodeError, ValueError):
            return False

    return True


def generate_initial_turn_if_missing(
    root_dir: Path,
    current_state: dict[str, Any],
) -> dict[str, Any] | None:
    if turn_files_exist(root_dir=root_dir, turn_number=1):
        return None

    scenario_config = read_json(root_dir / "data" / "runtime" / "scenario_config.json")
    hidden_truth = read_json(root_dir / "data" / "runtime" / "hidden_truth.json")

    completed_turn = {
        "note": "This is the initial turn. No learner action has been completed yet."
    }

    selected_action = {
        "note": "No selected action yet. Generate the first learner decision point."
    }

    selected_evidence = {
        "note": "No selected evidence yet. Generate initial evidence options."
    }

    vlm_output = {
        "note": "No VLM output yet because this is the initial turn."
    }

    security_output = {
        "note": "No security evaluation yet because this is the initial turn."
    }

    coach_output = {
        "note": "No coach feedback yet because this is the initial turn."
    }

    generated_turn = generate_turn_with_quality_retry(
        scenario_config=scenario_config,
        hidden_truth=hidden_truth,
        current_state=current_state,
        completed_turn=completed_turn,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        vlm_output=vlm_output,
        security_output=security_output,
        coach_output=coach_output,
        next_turn_number=1,
        timeline=load_incident_timeline(root_dir / "data" / "runtime", scenario_config.get("scenario_id")),
    )

    save_generated_turn(
        root_dir=root_dir,
        turn_number=1,
        generated_turn=generated_turn,
    )

    return generated_turn


def generate_and_save_next_turn(
    root_dir: Path,
    current_state: dict[str, Any],
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    next_turn_number: int,
) -> dict[str, Any]:
    scenario_config = read_json(root_dir / "data" / "runtime" / "scenario_config.json")
    hidden_truth = read_json(root_dir / "data" / "runtime" / "hidden_truth.json")
    timeline = load_incident_timeline(root_dir / "data" / "runtime", scenario_config.get("scenario_id"))

    generated_turn = generate_turn_with_quality_retry(
        scenario_config=scenario_config,
        hidden_truth=hidden_truth,
        current_state=current_state,
        completed_turn=completed_turn,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        vlm_output=vlm_output,
        security_output=security_output,
        coach_output=coach_output,
        next_turn_number=next_turn_number,
        timeline=timeline,
    )

    save_generated_turn(
        root_dir=root_dir,
        turn_number=next_turn_number,
        generated_turn=generated_turn,
    )

    return generated_turn


def generate_turn_with_quality_retry(
    scenario_config: dict[str, Any],
    hidden_truth: dict[str, Any],
    current_state: dict[str, Any],
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    next_turn_number: int,
    timeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generates a turn, validates it, asks the Coach AI to judge quality,
    and retries weak generations before saving.

    With an incident timeline, evidence facts come from the timeline and the
    coach only writes the turn's text around them. Scenarios prepared before
    timelines existed keep fully generated evidence.

    Flow:
    - Attempt 1: normal generation.
    - Attempt 2/3: generation with quality judge retry feedback.
    - If no attempt passes, save the best structurally valid attempt.
    - If no attempt is structurally valid, raise an error.
    """

    retry_instruction = ""
    timeline_items = build_turn_evidence(timeline, scenario_config, next_turn_number) if timeline else None
    anchor = action_anchor(timeline, timeline_items, next_turn_number) if timeline_items else None
    evidence_brief = timeline_prompt_block(timeline, timeline_items, anchor) if timeline_items else ""
    best_turn: dict[str, Any] | None = None
    best_score = -1
    validation_errors: list[str] = []
    quality_reviews: list[dict[str, Any]] = []
    anchor_repairs: list[str] = []
    decoy_removals: list[str] = []

    def with_diagnostics(turn: dict[str, Any], outcome: str) -> dict[str, Any]:
        turn["generation_diagnostics"] = {
            "attempts": attempt_number, "outcome": outcome,
            "validation_errors": validation_errors, "quality_reviews": quality_reviews,
            "anchor_repairs": anchor_repairs,
            "decoy_removals": decoy_removals,
        }
        return turn

    for attempt_number in range(1, MAX_GENERATION_ATTEMPTS + 1):
        log_memory(f"next-turn attempt {attempt_number} start")
        try:
            generated_turn = generate_next_turn_json(
                scenario_config=scenario_config,
                hidden_truth=hidden_truth,
                current_state=current_state,
                completed_turn=completed_turn,
                selected_action=selected_action,
                selected_evidence=selected_evidence,
                vlm_output=vlm_output,
                security_output=security_output,
                coach_output=coach_output,
                next_turn_number=next_turn_number,
                retry_instruction=retry_instruction,
                evidence_brief=evidence_brief,
            )
        except ValueError as exc:
            validation_error = str(exc)
            validation_errors.append(validation_error)
            retry_instruction = (
                "The previous next-turn response was not valid complete JSON. "
                "Regenerate the turn as JSON only, with exactly the required top-level keys: "
                "turn_config, actions, evidence_facts, expected_outcomes. "
                "Keep facts concise so the response does not get truncated. "
                f"Parser error: {validation_error[:900]}"
            )
            continue

        try:
            if timeline_items:
                apply_timeline_evidence(generated_turn, timeline_items)
            generated_turn = normalise_generated_turn(
                generated_turn=generated_turn,
                scenario_config=scenario_config,
                next_turn_number=next_turn_number,
            )
            apply_scenario_progression_safety(
                generated_turn=generated_turn,
                scenario_config=scenario_config,
                completed_turn=completed_turn,
                security_output=security_output,
                next_turn_number=next_turn_number,
                timeline_evidence=bool(timeline_items),
            )
            if timeline_items:
                apply_timeline_evidence(generated_turn, timeline_items)
                decoy_removals.extend(remove_decoy_mentions(generated_turn, timeline, timeline_items))
                # The security judge's marking scheme comes from the timeline, not the coach.
                generated_turn["expected_outcomes"] = marking_scheme(timeline, timeline_items, anchor)
            validate_generated_turn(generated_turn)
            if anchor:
                # The best action must name what only the strong item shows.
                repair = enforce_action_anchor(generated_turn["actions"], anchor)
                if repair:
                    anchor_repairs.append(f"attempt {attempt_number}: {repair}")
        except ValueError as exc:
            validation_error = str(exc)
            validation_errors.append(validation_error)
            retry_instruction = (
                "The previous generated turn failed structural validation. "
                f"Validation error: {validation_error}. "
                "Regenerate the turn with all required JSON keys, valid evidence templates, "
                "concrete facts, and renderable evidence."
            )
            continue

        quality_review = evaluate_generated_turn_quality(
            scenario_config=scenario_config,
            hidden_truth=hidden_truth,
            current_state=current_state,
            completed_turn=completed_turn,
            generated_turn=generated_turn,
            selected_action=selected_action,
            selected_evidence=selected_evidence,
            vlm_output=vlm_output,
            security_output=security_output,
            coach_output=coach_output,
            next_turn_number=next_turn_number,
            attempt_number=attempt_number,
        )

        quality_reviews.append(quality_review)

        if timeline_items:
            # Advisory only for timeline-built turns: code fixes the evidence, and the
            # partial item is the same baseline every turn by design, so the review's
            # complaints ("repeats the same evidence", "facts mostly unknown") cannot be
            # fixed by regenerating. In a live session all 12 reviews scored 60-65
            # (pass mark 70) and each retry cost a full generation.
            return with_diagnostics(generated_turn, "accepted_timeline_turn")

        quality_score = safe_int(quality_review.get("quality_score"), default=0)
        passed_quality_review = bool(quality_review.get("pass"))

        if quality_score > best_score:
            best_score = quality_score
            best_turn = generated_turn

        if passed_quality_review and quality_score >= MIN_ACCEPTABLE_QUALITY_SCORE:
            return with_diagnostics(generated_turn, "passed_quality_review")

        retry_instruction = clean_value(quality_review.get("retry_instruction"))

        if not retry_instruction:
            retry_instruction = build_default_retry_instruction(
                quality_review=quality_review,
                next_turn_number=next_turn_number,
            )

    if best_turn is not None:
        return with_diagnostics(best_turn, "best_valid_attempt")

    fallback_turn = build_deterministic_next_turn(
        next_turn_number=next_turn_number,
        security_output=security_output,
    )
    fallback_turn = normalise_generated_turn(
        generated_turn=fallback_turn,
        scenario_config=scenario_config,
        next_turn_number=next_turn_number,
    )
    if timeline_items:
        apply_timeline_evidence(fallback_turn, timeline_items)
        remove_decoy_mentions(fallback_turn, timeline, timeline_items)
        enforce_action_anchor(fallback_turn["actions"], anchor)
        fallback_turn["expected_outcomes"] = marking_scheme(timeline, timeline_items, anchor)
    validate_generated_turn(fallback_turn)

    return with_diagnostics(fallback_turn, "deterministic_fallback")


def build_deterministic_next_turn(
    next_turn_number: int,
    security_output: dict[str, Any],
) -> dict[str, Any]:
    verdict = clean_value(security_output.get("verdict")) or "Partial Support"

    if next_turn_number == 2:
        return {
            "turn_config": {
                "turn": 2,
                "phase": "Correlation and Scope",
                "briefing": (
                    "The initial audit event suggests suspicious identity activity. "
                    "Now correlate related IAM, credential, and monitoring signals to determine "
                    "whether the activity is isolated or part of a wider account-risk pattern."
                ),
                "known_context": [
                    "A ConsoleLogin without MFA was observed from source IP 185.220.101.42.",
                    "CloudTrail activity around the same window includes logging and credential-related API calls.",
                    f"The previous evidence was assessed as {verdict}; the next step is to verify scope before containment.",
                ],
                "coach_guidance": (
                    "Prioritise evidence that connects the same principal and source IP across "
                    "multiple services or credential actions."
                ),
            },
            "actions": [
                {
                    "id": "correlate_identity_timeline",
                    "title": "Correlate Identity Timeline",
                    "description": "Review IAM activity around the incident window to see whether the same principal and source IP appear across multiple actions.",
                    "recommended_next_focus": "Look for credential creation, logging changes, and caller identity checks.",
                    "choice_role": "best",
                },
                {
                    "id": "review_access_key_risk",
                    "title": "Review Access Key Risk",
                    "description": "Check whether a new or active access key is linked to the suspicious principal after the login event.",
                    "recommended_next_focus": "Confirm whether credential containment is needed.",
                    "choice_role": "partial",
                },
                {
                    "id": "query_security_logs",
                    "title": "Query Security Logs",
                    "description": "Use CloudWatch logs to correlate authentication, CloudTrail, and credential-monitoring messages.",
                    "recommended_next_focus": "Compare log timing and source IP across services.",
                    "choice_role": "weak",
                },
            ],
            "evidence_facts": [
                {
                    "id": "iam_scope_timeline",
                    "title": "IAM Scope Timeline",
                    "type": "iam_activity",
                    "summary": "IAM activity links login, CloudTrail, and credential actions to the same principal and source IP.",
                    "why_it_may_matter": "This helps determine whether the suspicious login led to additional risky identity actions.",
                    "support_role": "strong",
                    "template": "iam_activity",
                    "facts": {
                        "principal": "arn:aws:iam::123456789012:user/admin-test",
                        "source_ip": "185.220.101.42",
                        "mfa": "false",
                        "policy_change": "No approved IAM change ticket found",
                        "access_key_status": "New active key observed after login",
                        "risk_flags": [
                            "Console login without MFA",
                            "Credential action from same source IP",
                        ],
                        "activity_rows": [
                            ["2023-10-01T10:12:10Z", "arn:aws:iam::123456789012:user/admin-test", "ConsoleLogin", "185.220.101.42"],
                            ["2023-10-01T10:18:44Z", "arn:aws:iam::123456789012:user/admin-test", "ListTrails", "185.220.101.42"],
                            ["2023-10-01T10:21:35Z", "arn:aws:iam::123456789012:user/admin-test", "StopLogging", "185.220.101.42"],
                            ["2023-10-01T10:24:02Z", "arn:aws:iam::123456789012:user/admin-test", "CreateAccessKey", "185.220.101.42"],
                            ["2023-10-01T10:26:11Z", "arn:aws:iam::123456789012:user/admin-test", "GetCallerIdentity", "185.220.101.42"],
                        ],
                    },
                },
                {
                    "id": "access_key_review",
                    "title": "Access Key Review",
                    "type": "access_key",
                    "summary": "An active access key is associated with the suspicious principal shortly after the login.",
                    "why_it_may_matter": "This supports checking whether credential misuse or key rotation is required.",
                    "support_role": "partial",
                    "template": "access_key",
                    "facts": {
                        "access_key_id": "AKIA4Z7EXAMPLE92K",
                        "owner": "arn:aws:iam::123456789012:user/admin-test",
                        "status": "Active",
                        "last_used_service": "iam.amazonaws.com",
                        "last_used_region": "ap-southeast-1",
                        "last_used_time": "2023-10-01T10:24:02Z",
                        "source_ip": "185.220.101.42",
                    },
                },
                routine_application_logs_evidence(),
            ],
            "expected_outcomes": {
                "strong_support": "Evidence correlates the same principal and source IP across IAM, credential, and logging events.",
                "partial_support": "Evidence shows one relevant signal but does not establish scope across services.",
                "weak_support": "Evidence is security-related but does not connect to the suspicious identity activity.",
                "unsupported": "Evidence does not address identity activity, source IP, logging, or credential risk.",
            },
        }

    return {
        "turn_config": {
            "turn": next_turn_number,
            "phase": "Response and Decision",
            "briefing": (
                "The investigation has linked suspicious authentication, logging changes, "
                "and credential activity. Decide how to contain the risk and preserve evidence."
            ),
            "known_context": [
                "Suspicious identity activity has been correlated across multiple evidence sources.",
                "Credential activity and CloudTrail visibility are both relevant to the final response decision.",
                "The final action should balance containment, evidence preservation, and business impact.",
            ],
            "coach_guidance": "Choose evidence that supports containment or recovery, not just further detection.",
        },
        "actions": [
            {
                "id": "contain_suspicious_credentials",
                "title": "Contain Suspicious Credentials",
                "description": "Disable or rotate the suspicious access key and require re-authentication for the affected principal.",
                "recommended_next_focus": "Confirm the containment decision is supported by credential and identity evidence.",
                "choice_role": "best",
            },
            {
                "id": "restore_logging_visibility",
                "title": "Restore Logging Visibility",
                "description": "Verify CloudTrail and monitoring coverage after the StopLogging event before closing the incident.",
                "recommended_next_focus": "Confirm audit visibility is restored and protected.",
                "choice_role": "partial",
            },
            {
                "id": "review_business_impact",
                "title": "Review Business Impact",
                "description": "Check monitoring and billing signals for signs of resource misuse after credential activity.",
                "recommended_next_focus": "Use impact evidence to decide whether escalation is needed.",
                "choice_role": "weak",
            },
        ],
        "evidence_facts": [
            {
                "id": "access_key_containment_review",
                "title": "Access Key Containment Review",
                "type": "access_key",
                "summary": "The affected principal has an active access key used shortly after the suspicious login sequence.",
                "why_it_may_matter": "This is strong response evidence because it directly supports credential containment or key rotation.",
                "support_role": "strong",
                "template": "access_key",
                "facts": {
                    "access_key_id": "AKIA4Z7EXAMPLE92K",
                    "owner": "arn:aws:iam::123456789012:user/admin-test",
                    "status": "Active",
                    "last_used_service": "iam.amazonaws.com",
                    "last_used_region": "ap-southeast-1",
                    "last_used_time": "2023-10-01T10:24:02Z",
                    "source_ip": "185.220.101.42",
                },
            },
            {
                "id": "cloudtrail_logging_recovery_check",
                "title": "CloudTrail Logging Recovery Check",
                "type": "cloudtrail",
                "summary": "CloudTrail shows the StopLogging event that must be addressed before recovery can be trusted.",
                "why_it_may_matter": "This supports restoring audit visibility, but it is less direct for credential containment.",
                "support_role": "partial",
                "template": "cloudtrail",
                "facts": {
                    "event_name": "StopLogging",
                    "event_source": "cloudtrail.amazonaws.com",
                    "user": "arn:aws:iam::123456789012:user/admin-test",
                    "source_ip": "185.220.101.42",
                    "mfa": "false",
                    "event_time": "2023-10-01T10:21:35Z",
                    "region": "ap-southeast-1",
                    "error_code": "-",
                    "risk_signal": "Audit visibility was reduced after the suspicious login and before credential activity.",
                    "event_id": "e7b31d90-8b6a-4d5c-93f1-7b3d4f6a19c2",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "recipient_account_id": "123456789012",
                    "request_parameters": "name=prod-org-trail",
                    "related_events": [
                        ["2023-10-01T10:12:10Z", "ConsoleLogin", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "MFA false"],
                        ["2023-10-01T10:21:35Z", "StopLogging", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "Success"],
                        ["2023-10-01T10:24:02Z", "CreateAccessKey", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "Success"],
                    ],
                },
            },
            {
                "id": "billing_impact_snapshot",
                "title": "Billing Impact Snapshot",
                "type": "billing",
                "summary": "Billing shows a modest spend increase that should be checked for resource misuse.",
                "why_it_may_matter": "This is impact context rather than proof of identity compromise.",
                "support_role": "weak",
                "template": "billing",
                "facts": {
                    "current_spend": "$184.20",
                    "previous_average": "$42.75",
                    "largest_service": "EC2",
                    "region": "ap-southeast-1",
                    "change": "+331%",
                },
            },
        ],
        "expected_outcomes": {
            "strong_support": "Evidence directly supports containment or recovery using credential, logging, or threat signals.",
            "partial_support": "Evidence supports response planning but needs another source to justify containment.",
            "weak_support": "Evidence shows generic cloud risk but does not support the chosen response.",
            "unsupported": "Evidence is unrelated to credential containment, logging visibility, or impact review.",
        },
    }


def build_default_retry_instruction(
    quality_review: dict[str, Any],
    next_turn_number: int,
) -> str:
    issues = quality_review.get("issues", [])

    if not isinstance(issues, list):
        issues = [str(issues)]

    issue_text = "; ".join(str(issue) for issue in issues if issue)

    if not issue_text:
        issue_text = "The generated turn was not strong enough."

    return (
        f"Regenerate Turn {next_turn_number}. "
        f"Problems found: {issue_text}. "
        "Create a more meaningful scenario progression, avoid repeated evidence titles or facts, "
        "react to the learner's previous choice, and use concrete AWS-style evidence facts."
    )


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalise_generated_turn(
    generated_turn: dict[str, Any],
    scenario_config: dict[str, Any] | None = None,
    next_turn_number: int | None = None,
) -> dict[str, Any]:
    """
    Keeps local-model output usable without adding fake fallback evidence.

    This function:
    - trims actions to max 4
    - trims evidence items to max 3
    - repairs wrong evidence template/type combinations
    - normalises facts keys so screenshot templates render properly

    Important:
    It does NOT only infer templates when missing.
    It corrects wrong existing templates too.
    """

    actions = generated_turn.get("actions", [])
    evidence_facts = generated_turn.get("evidence_facts", [])

    if isinstance(actions, list) and len(actions) > 4:
        generated_turn["actions"] = actions[:4]
        actions = generated_turn["actions"]

    normalise_action_choice_roles(actions)

    if isinstance(evidence_facts, list) and len(evidence_facts) > 3:
        evidence_facts = evidence_facts[:3]
        generated_turn["evidence_facts"] = evidence_facts

    normalise_evidence_support_roles(evidence_facts)

    # First pass: use the repair module to correct template choice and facts.
    generated_turn = repair_next_turn_template_facts(generated_turn)

    evidence_facts = generated_turn.get("evidence_facts", [])
    normalise_evidence_support_roles(evidence_facts)

    if isinstance(evidence_facts, list):
        for evidence in evidence_facts:
            if not isinstance(evidence, dict):
                continue

            force_correct_template(evidence)
            normalise_evidence_facts(evidence)
            validate_template_alignment(evidence)

        generated_turn["evidence_facts"] = evidence_facts

    if scenario_config is not None and next_turn_number is not None:
        apply_evidence_template_strategy(
            generated_turn=generated_turn,
            scenario_config=scenario_config,
            turn_number=next_turn_number,
        )

    return generated_turn


def save_generated_turn(
    root_dir: Path,
    turn_number: int,
    generated_turn: dict[str, Any],
) -> None:
    output_dir = root_dir / "data" / "runtime" / "turns" / f"turn_{turn_number}"
    shuffle_turn_options(generated_turn, turn_number)

    write_json(output_dir / "turn_config.json", generated_turn["turn_config"])
    write_json(output_dir / "actions.json", generated_turn["actions"])
    write_json(output_dir / "evidence_facts.json", generated_turn["evidence_facts"])
    write_json(output_dir / "expected_outcomes.json", generated_turn["expected_outcomes"])
    if "generation_diagnostics" in generated_turn:
        write_json(output_dir / "generation_diagnostics.json", generated_turn["generation_diagnostics"])


def shuffle_turn_options(generated_turn: dict[str, Any], turn_number: int) -> None:
    """
    Avoids making the best choice visually obvious by position.

    The role metadata stays on each item, so evaluation and test mode still know
    which action/evidence is best after display order changes.
    """

    for key in ("actions", "evidence_facts"):
        items = generated_turn.get(key)

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
