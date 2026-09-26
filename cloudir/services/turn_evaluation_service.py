from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy
from typing import Any

from cloudir.ai_models.coach_model import unload_coach_model
from cloudir.ai_models.model_lifecycle import log_memory
from cloudir.scenario.turn_orchestrator import generate_and_save_next_turn
from cloudir.scenario.scenario_state import ScenarioState
from cloudir.services.evaluation_trace_service import (
    load_turn_evaluation_traces,
    save_turn_evaluation_trace,
)
from cloudir.services.final_debrief_service import (
    generate_and_save_final_debrief,
    get_final_debrief,
)


def prepare_next_turn_after_evaluation(
    root_dir: Path,
    scenario_state: ScenarioState,
    current_turn_number: int,
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    skip_next_turn: bool = False,
) -> dict[str, Any]:
    """Generates the next turn unless the learner has completed the final turn."""

    if current_turn_number >= scenario_state.max_turns:
        scenario_state.pending_turn = current_turn_number

        return {
            "isComplete": True,
            "nextTurn": current_turn_number,
            "message": "Prototype scenario complete. Final containment decision reached.",
        }

    if skip_next_turn:
        # Test Mode only: judge this turn's evidence without spending minutes
        # generating a turn that will not be played. The run cannot continue.
        scenario_state.pending_turn = None

        return {
            "isComplete": False,
            "skipped": True,
            "nextTurn": None,
            "message": (
                "Test Mode: next-turn generation was skipped, so this run cannot continue. "
                "Use Retry from a checkpoint or start a new run to test another case."
            ),
        }

    next_turn_number = current_turn_number + 1

    generated_turn = generate_and_save_next_turn(
        root_dir=root_dir,
        current_state=scenario_state.to_dict(),
        completed_turn=completed_turn,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        vlm_output=vlm_output,
        security_output=security_output,
        coach_output=coach_output,
        next_turn_number=next_turn_number,
    )

    scenario_state.pending_turn = next_turn_number

    return {
        "isComplete": False,
        "nextTurn": next_turn_number,
        "message": f"Turn {next_turn_number} has been generated in the background.",
        "nextBriefing": generated_turn["turn_config"]["briefing"],
        "nextPhase": generated_turn["turn_config"]["phase"],
    }


def authored_action_role(root_dir: Path, turn_number: int, selected_action: dict[str, Any]) -> str | None:
    """The choice role the turn's author gave the selected action, read from the turn's
    own files rather than the browser request, so a request cannot claim "best"."""
    path = root_dir / "data" / "runtime" / "turns" / f"turn_{turn_number}" / "actions.json"
    try:
        actions = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for action in actions if isinstance(actions, list) else []:
        if (selected_action.get("id") and action.get("id") == selected_action.get("id")) or \
                (not selected_action.get("id") and action.get("title") == selected_action.get("title")):
            return action.get("choice_role")
    return None


def finish_evaluation(
    root_dir: Path,
    scenario_state: ScenarioState,
    current_turn_number: int,
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    learner_justification: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    skip_next_turn: bool = False,
) -> dict[str, Any]:
    """
    Shared tail for both /api/evaluate and /api/evaluate-stream: records the
    completed action, generates the next turn, and assembles the response
    payload both routes send back to the client.
    """

    if scenario_state.completed or scenario_state.has_recorded_turn(current_turn_number):
        is_complete = scenario_state.completed or current_turn_number >= scenario_state.max_turns
        scenario_state.normalise_action_history()
        final_debrief = get_final_debrief(scenario_state.to_dict()) if is_complete else {}

        return {
            "aiResult": {
                "vlm_output": vlm_output,
                "security_output": security_output,
                "coach_output": coach_output,
            },
            "turnTrace": {},
            "evaluationTraces": load_turn_evaluation_traces(),
            "finalDebrief": final_debrief,
            "nextTurnPreview": {
                "isComplete": is_complete,
                "nextTurn": scenario_state.pending_turn or scenario_state.current_turn,
                "message": (
                    "Scenario is already complete. Final coach debrief is ready."
                    if is_complete
                    else "This turn was already evaluated. Continue to the prepared next turn."
                ),
            },
            "state": scenario_state.to_dict(),
        }

    verdict = security_output.get("verdict", "Unknown")
    state_before_turn = deepcopy(scenario_state.to_dict())

    scenario_state.record_action(
        action_title=selected_action.get("title", "Unknown action"),
        evidence_title=selected_evidence.get("title", "Unknown evidence"),
        verdict=verdict,
        action_role=authored_action_role(root_dir, current_turn_number, selected_action),
    )

    turn_trace = save_turn_evaluation_trace(
        turn_number=current_turn_number,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        learner_justification=learner_justification,
        vlm_output=vlm_output,
        security_output=security_output,
        coach_output=coach_output,
        state_after_turn=scenario_state.to_dict(),
        state_before_turn=state_before_turn,
    )

    # Next-turn generation and the final debrief reload the coach model; release
    # it afterwards so it is not left in memory for the next evaluation.
    try:
        next_turn_preview = prepare_next_turn_after_evaluation(
            root_dir=root_dir,
            scenario_state=scenario_state,
            current_turn_number=current_turn_number,
            completed_turn=completed_turn,
            selected_action=selected_action,
            selected_evidence=selected_evidence,
            vlm_output=vlm_output,
            security_output=security_output,
            coach_output=coach_output,
            skip_next_turn=skip_next_turn,
        )

        final_debrief = {}

        if next_turn_preview.get("isComplete") is True:
            scenario_state.completed = True
            scenario_state.normalise_action_history()
            final_debrief = generate_and_save_final_debrief(scenario_state.to_dict())
    finally:
        unload_coach_model()
        log_memory("after next turn / debrief")

    return {
        "aiResult": {
            "vlm_output": vlm_output,
            "security_output": security_output,
            "coach_output": coach_output,
        },
        "turnTrace": turn_trace,
        "evaluationTraces": load_turn_evaluation_traces(),
        "finalDebrief": final_debrief,
        "nextTurnPreview": next_turn_preview,
        "state": scenario_state.to_dict(),
    }
