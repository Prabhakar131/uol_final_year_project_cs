from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cloudir.paths import RUNTIME_DATA_DIR

from .select_case import processed_dir, scenario_slug, source_assets_dir


SCENARIO_DATA_DIR = RUNTIME_DATA_DIR


def build_runtime_files(scenario_id: str = "identity-management") -> dict[str, Any]:
    """
    Converts prepared ACSE files into CloudIR runtime files.

    No AI is used in this stage.
    This file does not invent new security meaning. It packages the selected
    ACSE case, VLM architecture facts, and security-model threat normalisation.
    """

    slug = scenario_slug(scenario_id)
    input_dir = processed_dir(scenario_id)

    selected_case = _read_json(input_dir / "selected_acse_case.json")
    architecture_facts = _read_json(input_dir / "architecture_facts.json")
    normalised_threat_model = _read_json(input_dir / "normalised_threat_model.json")
    source_dir = source_assets_dir(scenario_id)
    source_dir_relative = source_dir.relative_to(source_dir.parents[2])
    scenario_title = format_scenario_title(scenario_id)

    scenario_seed = {
        "dataset": "ACSE-Eval",
        "scenario_id": scenario_id,
        "scenario_slug": slug,
        "scenario_focus": scenario_id,
        "selected_acse_case": selected_case,
        "architecture_facts": architecture_facts,
        "normalised_threat_model": normalised_threat_model,
        "learner_starting_context": {
            "source": "Prepared from ACSE-Eval dataset inputs.",
            "architecture_summary": architecture_facts.get("architecture_summary", ""),
            "primary_risk": normalised_threat_model.get("primary_risk", ""),
            "investigation_goals": normalised_threat_model.get(
                "investigation_goals", []
            ),
            "recommended_evidence_sources": normalised_threat_model.get(
                "recommended_evidence_sources", []
            ),
        },
    }

    hidden_truth = {
        "dataset": "ACSE-Eval",
        "scenario_id": scenario_id,
        "scenario_slug": slug,
        "scenario_type": normalised_threat_model.get("scenario_type", scenario_id),
        "primary_risk": normalised_threat_model.get("primary_risk", ""),
        "affected_assets": normalised_threat_model.get("affected_assets", []),
        "likely_attack_path": normalised_threat_model.get("likely_attack_path", []),
        "security_signals": normalised_threat_model.get("security_signals", []),
        "hidden_truth_candidates": normalised_threat_model.get(
            "hidden_truth_candidates", []
        ),
        "architecture_summary": architecture_facts.get("architecture_summary", ""),
    }

    scenario_config = {
        "dataset": "ACSE-Eval",
        "scenario_id": scenario_id,
        "scenario_slug": slug,
        "scenario_title": f"ACSE-Eval {scenario_title} Incident",
        "total_turns": 5,
        "starting_turn": 1,
        "allowed_evidence_templates": [
            "cloudtrail",
            "iam_activity",
            "guardduty",
            "cloudwatch",
            "access_key",
            "billing",
        ],
        "evidence_template_strategy": build_evidence_template_strategy(scenario_id),
        "dataset_context": {
            "architecture_asset": (
                f"{source_dir_relative}/architecture.png"
            ),
            "threat_model_asset": (
                f"{source_dir_relative}/threat-model.json"
            ),
        },
        "scenario_progression": {
            "progression_mode": "ai_guided_adaptive",
            "turn_strategy": (
                "Each turn should adapt to the learner's action, selected evidence, "
                "security verdict, and coach feedback."
            ),
            "incident_response_stages": [
                {
                    "turn": 1,
                    "stage": "detection_and_initial_triage",
                    "goal": (
                        "Help the learner identify the first identity-related "
                        "security signal and choose useful initial evidence."
                    ),
                },
                {
                    "turn": 2,
                    "stage": "correlation_and_scope",
                    "goal": (
                        "Help the learner correlate identity activity, affected "
                        "assets, credential risk, or logging gaps."
                    ),
                },
                {
                    "turn": 3,
                    "stage": "scope_and_blast_radius",
                    "goal": (
                        "Help the learner decide how far the incident may have "
                        "spread across credentials, workloads, logs, or cost impact."
                    ),
                },
                {
                    "turn": 4,
                    "stage": "containment_and_remediation",
                    "goal": (
                        "Help the learner choose a proportionate containment or "
                        "remediation step using direct supporting evidence."
                    ),
                },
                {
                    "turn": 5,
                    "stage": "recovery_and_final_debrief",
                    "goal": (
                        "Help the learner verify recovery, residual risk, and final "
                        "incident reporting evidence before closing the scenario."
                    ),
                },
            ],
            "adaptation_rules": [
                "Strong evidence use should progress the scenario.",
                "Partial evidence use should progress with uncertainty.",
                "Weak or unsupported evidence use should refocus the learner.",
                "Hidden truth must not be directly exposed to the learner.",
            ],
        },
    }

    SCENARIO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    _write_json(SCENARIO_DATA_DIR / "scenario_seed.json", scenario_seed)
    _write_json(SCENARIO_DATA_DIR / "hidden_truth.json", hidden_truth)
    _write_json(SCENARIO_DATA_DIR / "scenario_config.json", scenario_config)

    return {
        "scenario_seed": scenario_seed,
        "hidden_truth": hidden_truth,
        "scenario_config": scenario_config,
    }


def format_scenario_title(scenario_id: str) -> str:
    return scenario_id.strip().replace("_", "-").replace("-", " ").title()


def build_evidence_template_strategy(scenario_id: str) -> dict[str, Any]:
    scenario_key = scenario_slug(scenario_id).replace("_", "-")

    strategies: dict[str, dict[str, dict[str, str]]] = {
        "automated-security-response": {
            "1": {
                "strong": "guardduty",
                "partial": "cloudtrail",
                "weak": "access_key",
            },
            "2": {
                "strong": "cloudwatch",
                "partial": "cloudtrail",
                "weak": "access_key",
            },
            "3": {
                "strong": "cloudtrail",
                "partial": "cloudwatch",
                "weak": "billing",
            },
            "4": {
                "strong": "access_key",
                "partial": "guardduty",
                "weak": "billing",
            },
            "5": {
                "strong": "cloudwatch",
                "partial": "cloudtrail",
                "weak": "billing",
            },
        },
        "identity-management": {
            "1": {
                "strong": "cloudtrail",
                "partial": "iam_activity",
                "weak": "billing",
            },
            "2": {
                "strong": "iam_activity",
                "partial": "access_key",
                "weak": "cloudwatch",
            },
            "3": {
                "strong": "guardduty",
                "partial": "cloudtrail",
                "weak": "billing",
            },
            "4": {
                "strong": "access_key",
                "partial": "cloudwatch",
                "weak": "billing",
            },
            "5": {
                "strong": "cloudtrail",
                "partial": "iam_activity",
                "weak": "billing",
            },
        },
        "cost-management": {
            "1": {
                "strong": "billing",
                "partial": "cloudwatch",
                "weak": "cloudtrail",
            },
            "2": {
                "strong": "cloudwatch",
                "partial": "billing",
                "weak": "guardduty",
            },
            "3": {
                "strong": "billing",
                "partial": "billing",
                "weak": "access_key",
            },
            "4": {
                "strong": "cloudtrail",
                "partial": "guardduty",
                "weak": "access_key",
            },
            "5": {
                "strong": "cloudwatch",
                "partial": "billing",
                "weak": "iam_activity",
            },
        },
    }

    role_templates_by_turn = strategies.get(
        scenario_key,
        {
            "1": {
                "strong": "cloudtrail",
                "partial": "cloudwatch",
                "weak": "billing",
            },
            "2": {
                "strong": "cloudwatch",
                "partial": "cloudtrail",
                "weak": "guardduty",
            },
            "3": {
                "strong": "access_key",
                "partial": "cloudwatch",
                "weak": "billing",
            },
            "4": {
                "strong": "guardduty",
                "partial": "cloudtrail",
                "weak": "billing",
            },
            "5": {
                "strong": "cloudwatch",
                "partial": "iam_activity",
                "weak": "billing",
            },
        },
    )

    return {
        "mode": "scenario_turn_roles",
        "description": (
            "Assign strong, partial, and weak evidence roles by scenario and turn "
            "so the best evidence source changes with the investigation stage."
        ),
        "role_templates_by_turn": role_templates_by_turn,
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required prepared file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
