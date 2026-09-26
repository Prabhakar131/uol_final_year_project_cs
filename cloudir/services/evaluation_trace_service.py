from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cloudir.paths import RUNTIME_DATA_DIR
from cloudir.scenario.json_boundary import to_camel_case_keys


EVALUATIONS_DIR = RUNTIME_DATA_DIR / "evaluations"
FINAL_DEBRIEF_PATH = EVALUATIONS_DIR / "final_debrief.json"


def save_turn_evaluation_trace(
    *,
    turn_number: int,
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    learner_justification: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    state_after_turn: dict[str, Any],
    state_before_turn: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace = {
        "turn": turn_number,
        "selected_action": selected_action,
        "selected_evidence": selected_evidence,
        "learner_justification": learner_justification,
        "vlm_output": vlm_output,
        "security_output": security_output,
        "coach_output": coach_output,
        "state_after_turn": state_after_turn,
        "state_before_turn": state_before_turn,
    }

    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(turn_trace_path(turn_number), trace)

    return trace


def load_turn_evaluation_traces() -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []

    if not EVALUATIONS_DIR.exists():
        return traces

    for path in sorted(EVALUATIONS_DIR.glob("turn_*_evaluation.json")):
        payload = read_json(path)

        if isinstance(payload, dict):
            traces.append(payload)

    return traces


def save_final_debrief(debrief: dict[str, Any]) -> None:
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(FINAL_DEBRIEF_PATH, debrief)


def load_final_debrief() -> dict[str, Any]:
    if not FINAL_DEBRIEF_PATH.exists():
        return {}

    payload = read_json(FINAL_DEBRIEF_PATH)

    return to_camel_case_keys(payload) if isinstance(payload, dict) else {}


def clear_evaluation_traces() -> None:
    if EVALUATIONS_DIR.exists():
        shutil.rmtree(EVALUATIONS_DIR)

    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)


def turn_trace_path(turn_number: int) -> Path:
    return EVALUATIONS_DIR / f"turn_{turn_number}_evaluation.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
