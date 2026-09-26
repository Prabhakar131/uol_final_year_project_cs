from __future__ import annotations

from typing import Any

from cloudir.evidence_core.template_inference import force_correct_template, validate_template_alignment
from cloudir.evidence_core.fact_repair import normalise_evidence_facts
from cloudir.evidence_core.evidence_text_helpers import clean_value
from cloudir.scenario.action_choice_roles import normalise_action_choice_roles
from cloudir.scenario.evidence_support_roles import normalise_evidence_support_roles
from cloudir.scenario.evidence_strength import validate_support_role_content


def validate_generated_turn(generated_turn: dict[str, Any]) -> None:
    required_keys = {
        "turn_config",
        "actions",
        "evidence_facts",
        "expected_outcomes",
    }

    missing = required_keys - set(generated_turn.keys())

    if missing:
        raise ValueError(f"Generated turn is missing keys: {missing}")

    validate_turn_config(generated_turn["turn_config"])
    validate_actions(generated_turn["actions"])
    validate_evidence_facts(generated_turn["evidence_facts"])
    validate_expected_outcomes(generated_turn["expected_outcomes"])


def validate_turn_config(turn_config: dict[str, Any]) -> None:
    if not isinstance(turn_config, dict):
        raise ValueError("turn_config must be a JSON object.")

    required_keys = {
        "turn",
        "phase",
        "briefing",
        "known_context",
        "coach_guidance",
    }

    missing = required_keys - set(turn_config.keys())

    if missing:
        raise ValueError(f"turn_config is missing keys: {missing}")

    if not isinstance(turn_config["known_context"], list):
        raise ValueError("turn_config.known_context must be a list.")


def validate_actions(actions: Any) -> None:
    if not isinstance(actions, list):
        raise ValueError("Generated turn actions must be a list.")

    if len(actions) < 1:
        raise ValueError("Generated turn must contain at least 1 action.")

    if len(actions) > 4:
        raise ValueError(
            "Generated turn cannot contain more than 4 actions after normalisation."
        )

    required_action_keys = {
        "id",
        "title",
        "description",
        "recommended_next_focus",
        "choice_role",
    }

    normalise_action_choice_roles(actions)

    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("Each action must be a JSON object.")

        missing = required_action_keys - set(action.keys())

        if missing:
            raise ValueError(
                f"Action {action.get('id', 'unknown')} is missing keys: {missing}"
            )


def validate_evidence_facts(evidence_facts: Any) -> None:
    if not isinstance(evidence_facts, list):
        raise ValueError("Generated turn evidence_facts must be a list.")

    if len(evidence_facts) < 1:
        raise ValueError("Generated turn must contain at least 1 evidence item.")

    if len(evidence_facts) > 3:
        raise ValueError(
            "Generated turn cannot contain more than 3 evidence items after normalisation."
        )

    normalise_evidence_support_roles(evidence_facts)

    roles = [item.get("support_role") for item in evidence_facts if isinstance(item, dict)]
    if any(role not in {"strong", "partial", "weak"} for role in roles):
        raise ValueError("Evidence needs explicit support_role labels justified by its facts; do not assign by position.")
    if len(evidence_facts) == 3 and set(roles) != {"strong", "partial", "weak"}:
        raise ValueError("Regenerate evidence with strong, partial and weak content; changing labels alone is not a repair.")

    required_evidence_keys = {
        "id",
        "title",
        "type",
        "summary",
        "why_it_may_matter",
        "support_role",
        "template",
        "facts",
    }

    used_templates: set[str] = set()

    for evidence in evidence_facts:
        if not isinstance(evidence, dict):
            raise ValueError("Each evidence item must be a JSON object.")

        missing = required_evidence_keys - set(evidence.keys())

        if missing:
            raise ValueError(
                f"Evidence item {evidence.get('id', 'unknown')} is missing keys: {missing}"
            )

        force_correct_template(evidence)
        normalise_evidence_facts(evidence)
        validate_template_alignment(evidence)

        if not isinstance(evidence["facts"], dict):
            raise ValueError(
                f"Evidence item {evidence.get('id')} has invalid facts field. It must be a JSON object."
            )

        template = clean_value(evidence.get("template"))

        if template in used_templates:
            raise ValueError(
                f"Duplicate evidence template detected in the same turn: {template}. "
                "Each generated evidence item should use a different template so the learner receives distinct evidence sources."
            )

        used_templates.add(template)

    # Runs after normalisation so it checks what the screenshot will render.
    validate_support_role_content(evidence_facts)


def validate_expected_outcomes(expected_outcomes: Any) -> None:
    if not isinstance(expected_outcomes, dict):
        raise ValueError("expected_outcomes must be a JSON object.")

    required_keys = {
        "strong_support",
        "partial_support",
        "weak_support",
    }

    missing = required_keys - set(expected_outcomes.keys())

    if missing:
        raise ValueError(f"expected_outcomes is missing keys: {missing}")
