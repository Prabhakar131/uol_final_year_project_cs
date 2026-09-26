"""
Retry ONLY case 3 (cloudtrail_event, expected weak) of the weak-evidence experiment.
Cases 1-2 already saved successfully; this appends case 3 to the same results file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cloudir.ai_models.run_turn import run_ai_evaluation_turn  # noqa: E402
from cloudir.ai_models.security_model import security_model_label  # noqa: E402
from cloudir.scenario.scenario_state import ScenarioState  # noqa: E402

TURN_DIR = ROOT / "data" / "runtime" / "turns" / "turn_1"
EVIDENCE_DIR = ROOT / "output" / "generated_evidence" / "turn_1"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "experiments" / "learner_evidence_choice_experiment.json"


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    actions = load_json(TURN_DIR / "actions.json")
    evidence_facts = load_json(TURN_DIR / "evidence_facts.json")
    selected_action = next(a for a in actions if a["id"] == "review_iam_activities")

    evidence_id = "cloudtrail_event"
    evidence = dict(next(e for e in evidence_facts if e["id"] == evidence_id))
    image_path = EVIDENCE_DIR / "cloudtrail_event.png"
    assert image_path.exists()
    evidence["imagePath"] = str(image_path)

    justification = {
        "typed_text": (
            f"I selected {evidence['title']} because it may be relevant to reviewing "
            "recent IAM activity for this incident."
        ),
        "source": "typed_justification",
    }

    state_before = ScenarioState().to_dict()

    print(f"=== Running case: {evidence_id} (expected support_role={evidence['support_role']}) ===")
    started = time.perf_counter()
    try:
        pipeline_output = run_ai_evaluation_turn(
            selected_action=selected_action,
            selected_evidence=evidence,
            scenario_state=state_before,
            root_dir=ROOT,
            learner_justification=justification,
        )
        error = None
    except Exception as exc:  # noqa: BLE001
        pipeline_output = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    print(f"  elapsed: {elapsed:.1f}s  error={error}")

    state_before_snake = {
        "current_turn": state_before.get("currentTurn", 1),
        "max_turns": state_before.get("maxTurns", 5),
        "containment": state_before["containment"],
        "visibility": state_before["visibility"],
        "risk": state_before["risk"],
        "phase": state_before["phase"],
        "action_history": state_before.get("actionHistory", []),
        "pending_turn": state_before.get("pendingTurn"),
        "completed": state_before.get("completed", False),
    }
    state_obj = ScenarioState(**state_before_snake)

    verdict = None
    if pipeline_output:
        security_output = pipeline_output.get("security_output") or {}
        verdict = security_output.get("verdict")
        if verdict:
            state_obj.record_action(
                action_title=selected_action["title"],
                evidence_title=evidence["title"],
                verdict=verdict,
            )

    state_after = state_obj.to_dict()

    result = {
        "scenario_id": "cost-management",
        "turn": 1,
        "selected_action": {
            "id": selected_action["id"],
            "title": selected_action["title"],
            "choice_role": selected_action["choice_role"],
        },
        "selected_evidence": {
            "id": evidence["id"],
            "title": evidence["title"],
            "template": evidence["template"],
            "expected_support_role": evidence["support_role"],
        },
        "learner_justification": justification["typed_text"],
        "model_config": {
            "vlm": "Qwen2.5-VL-3B-Instruct",
            "security": security_model_label(),
            "coach": "Qwen2.5-1.5B-Instruct",
        },
        "elapsed_seconds": round(elapsed, 1),
        "error": error,
        "vlm_output": pipeline_output.get("vlm_output") if pipeline_output else None,
        "security_output": pipeline_output.get("security_output") if pipeline_output else None,
        "coach_output": pipeline_output.get("coach_output") if pipeline_output else None,
        "state_before": state_before,
        "state_after": state_after,
    }

    existing = json.loads(RESULTS_PATH.read_text())
    # remove any prior failed/partial attempt at this evidence_id, then append the fresh one
    existing["cases"] = [c for c in existing["cases"] if c["selected_evidence"]["id"] != evidence_id]
    existing["cases"].append(result)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2))
    print(f"saved. total cases now: {len(existing['cases'])}")
    print("Done.")


if __name__ == "__main__":
    main()
