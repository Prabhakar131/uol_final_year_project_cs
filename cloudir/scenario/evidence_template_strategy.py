from __future__ import annotations

from typing import Any


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
