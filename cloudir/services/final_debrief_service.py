from __future__ import annotations

from typing import Any

from cloudir.ai_models.coach_model import (
    build_fallback_final_debrief,
    generate_final_debrief_json,
)
from cloudir.services.evaluation_trace_service import (
    load_final_debrief,
    load_turn_evaluation_traces,
    save_final_debrief,
)


def generate_and_save_final_debrief(final_state: dict[str, Any]) -> dict[str, Any]:
    traces = load_turn_evaluation_traces()

    if not traces:
        debrief = build_fallback_final_debrief(
            evaluation_traces=[],
            final_state=final_state,
            error="No evaluation traces were available.",
        )
        save_final_debrief(debrief)
        return debrief

    try:
        debrief = generate_final_debrief_json(
            evaluation_traces=traces,
            final_state=final_state,
        )
    except Exception as exc:
        debrief = build_fallback_final_debrief(
            evaluation_traces=traces,
            final_state=final_state,
            error=str(exc),
        )

    save_final_debrief(debrief)

    return debrief


def get_final_debrief(final_state: dict[str, Any] | None = None) -> dict[str, Any]:
    debrief = load_final_debrief()

    if debrief:
        return debrief

    return build_fallback_final_debrief(
        evaluation_traces=load_turn_evaluation_traces(),
        final_state=final_state or {},
        error="Final debrief has not been generated yet.",
    )
