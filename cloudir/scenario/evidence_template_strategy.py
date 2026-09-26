from __future__ import annotations

from typing import Any

from cloudir.evidence_core.evidence_text_helpers import clean_value


SUPPORT_ROLES = ("strong", "partial", "weak")


def apply_evidence_template_strategy(
    generated_turn: dict[str, Any],
    scenario_config: dict[str, Any],
    turn_number: int,
) -> None:
    """Keep authored labels intact; template preferences belong in the prompt.

    A CloudWatch template can contain strong, partial or weak evidence depending
    on its actual rows and the selected action. Changing labels cannot implement
    the configured strategy after generation.
    """
    from cloudir.scenario.evidence_support_roles import normalise_evidence_support_roles

    normalise_evidence_support_roles(generated_turn.get("evidence_facts"))


def get_turn_evidence_strategy(
    scenario_config: dict[str, Any],
    turn_number: int,
) -> dict[str, str]:
    strategy = scenario_config.get("evidence_template_strategy")

    if not isinstance(strategy, dict):
        return {}

    role_templates = strategy.get("role_templates_by_turn")

    if not isinstance(role_templates, dict):
        return {}

    turn_strategy = role_templates.get(str(turn_number)) or role_templates.get(turn_number)

    if not isinstance(turn_strategy, dict):
        return {}

    return {
        role: clean_value(turn_strategy.get(role)).lower()
        for role in SUPPORT_ROLES
        if clean_value(turn_strategy.get(role))
    }


def find_unassigned_template_index(
    evidence_items: list[Any],
    template: str,
    assigned_indexes: set[int],
) -> int | None:
    for index, item in enumerate(evidence_items):
        if index in assigned_indexes or not isinstance(item, dict):
            continue

        item_template = clean_value(item.get("template") or item.get("type")).lower()

        if item_template == template:
            return index

    return None
