"""
Controlled learner evidence-choice experiment on the identity-management ACSE-Eval
scenario (Issue 7 replacement for the Issue 6 run, which was mistakenly executed
against cost-management runtime data instead of identity-management).

Same experimental logic as Issue 6: same scenario + same turn + same learner action
(review_iam_activities), only the selected evidence varies across its three naturally
occurring support roles (strong / partial / weak), as produced by the real dataset
pipeline (not manufactured). Scenario state is reset to a fresh ScenarioState() before
each case, so state deltas are directly comparable. Justification is typed only, no
STT, per Part B3. Uses the real production pipeline (cloudir.ai_models.run_turn.
run_ai_evaluation_turn) against the currently loaded identity-management runtime
scenario, with the production models configured in .env (Qwen2.5-VL-3B-Instruct,
Foundation-Sec-8B-Instruct, Qwen2.5-1.5B-Instruct). No models are substituted and no
results are invented. Cases run strictly sequentially (one model pipeline at a time)
for memory safety, with incremental save after each case.
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
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "experiments" / "learner_evidence_choice_experiment_identity_management.json"

IMAGE_MAP = {
    "cloudtrail_log_row": "cloudtrail_log_row.png",
    "cloudwatch_insights_query": "cloudwatch_insights_query.png",
    "access_key_info": "access_key_info.png",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    scenario_config = load_json(ROOT / "data" / "runtime" / "scenario_config.json")
    assert scenario_config.get("scenario_id") == "identity-management", (
        f"Runtime scenario is {scenario_config.get('scenario_id')!r}, expected "
        "'identity-management'. Rebuild the scenario before running this experiment."
    )

    actions = load_json(TURN_DIR / "actions.json")
    evidence_facts = load_json(TURN_DIR / "evidence_facts.json")

    # Fixed action for all cases, per Part B2: hold action constant, vary evidence only.
    selected_action = next(a for a in actions if a["id"] == "review_iam_activities")
    assert selected_action["choice_role"] == "best"

    # Ordered by expected support role (strong -> partial -> weak) as produced by the
    # scenario pipeline itself, not manufactured for this experiment.
    cases_order = ["cloudtrail_log_row", "cloudwatch_insights_query", "access_key_info"]
    evidence_by_id = {e["id"]: e for e in evidence_facts}

    results = []

    for evidence_id in cases_order:
        evidence = dict(evidence_by_id[evidence_id])
        image_path = EVIDENCE_DIR / IMAGE_MAP[evidence_id]
        assert image_path.exists(), f"missing generated evidence image: {image_path}"
        evidence["imagePath"] = str(image_path)

        justification = {
            "typed_text": (
                f"I selected {evidence['title']} because it may be relevant to reviewing "
                "recent IAM activity for this incident."
            ),
            "source": "typed_justification",
        }

        # Fresh default scenario state for every case, so state deltas are comparable
        # against the same starting point (isolates the effect of evidence choice).
        state_before = ScenarioState().to_dict()

        print(f"\n=== Running case: {evidence_id} (expected support_role={evidence['support_role']}) ===")
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

        # camelCase -> snake_case for reconstructing state object cleanly
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
            "scenario_id": "identity-management",
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
            "containment_delta": state_after["containment"] - state_before["containment"],
            "visibility_delta": state_after["visibility"] - state_before["visibility"],
        }
        results.append(result)

        # Save incrementally so partial progress is never lost.
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "experiment": "learner_evidence_choice_behaviour_identity_management",
            "description": (
                "Same action (review_iam_activities), same starting scenario state, three "
                "different evidence choices of differing expected support role, run through "
                "the real production pipeline (VLM -> security reasoning -> coach -> "
                "deterministic state update) against the identity-management ACSE-Eval "
                "scenario (replaces the Issue 6 cost-management run, which used the wrong "
                "runtime scenario)."
            ),
            "cases": results,
        }, indent=2))
        print(f"  saved progress to {RESULTS_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
