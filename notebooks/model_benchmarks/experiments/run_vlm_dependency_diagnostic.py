"""
Part D diagnostic: is the security-reasoning model's verdict dependent on the VLM's
own preliminary support/relevance judgement, rather than reasoning independently from
the underlying visible facts?

Uses the real saved identity-management weak case (access_key_info, expected weak
support) from learner_evidence_choice_experiment_identity_management.json, which
returned "Strong Support" from the security model despite being authored as weak.

Diagnostic A (current production input): calls evaluate_action_evidence with the
VLM output exactly as production passes it, including the VLM's own generated
"security_relevance" and "supports_selected_action" judgement fields.

Diagnostic B (facts-only input): calls evaluate_action_evidence with the SAME VLM
output, but with ONLY "security_relevance" and "supports_selected_action" removed.
Every other field (visible_facts_extracted, visible_evidence_summary,
important_visible_fields, reason, not_proven_by_this_evidence, evidence_type) is left
exactly as generated, unchanged.

This isolates one variable: whether the security model's verdict changes when the
VLM's own judgement fields are absent, while it still has all the same underlying
visible facts. Uses the real production security model (Foundation-Sec-8B-Instruct).
No fabricated results; both calls go through the same evaluate_action_evidence
function used in production.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cloudir.ai_models.security_model import evaluate_action_evidence  # noqa: E402
from cloudir.scenario.scenario_state import ScenarioState  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "experiments" / "learner_evidence_choice_experiment_identity_management.json"
OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "experiments" / "vlm_dependency_diagnostic_identity_management.json"

TURN_DIR = ROOT / "data" / "runtime" / "turns" / "turn_1"


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    experiment = load_json(RESULTS_PATH)
    case = next(c for c in experiment["cases"] if c["selected_evidence"]["id"] == "access_key_info")

    actions = load_json(TURN_DIR / "actions.json")
    evidence_facts = load_json(TURN_DIR / "evidence_facts.json")
    selected_action = next(a for a in actions if a["id"] == "review_iam_activities")
    selected_evidence = next(e for e in evidence_facts if e["id"] == "access_key_info")

    vlm_output_a = case["vlm_output"]  # exactly what production generated for this case
    assert "security_relevance" in vlm_output_a and "supports_selected_action" in vlm_output_a

    vlm_output_b = copy.deepcopy(vlm_output_a)
    del vlm_output_b["security_relevance"]
    del vlm_output_b["supports_selected_action"]

    justification = {
        "typed_text": case["learner_justification"],
        "source": "typed_justification",
    }
    scenario_state = ScenarioState().to_dict()

    print("=== Diagnostic A: current production input (VLM judgement included) ===")
    started = time.perf_counter()
    output_a = evaluate_action_evidence(
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        learner_justification=justification,
        vlm_output=vlm_output_a,
        scenario_state=scenario_state,
    )
    elapsed_a = time.perf_counter() - started
    print(f"  verdict: {output_a['verdict']}  ({elapsed_a:.1f}s)")

    print("\n=== Diagnostic B: facts-only input (VLM judgement fields removed) ===")
    started = time.perf_counter()
    output_b = evaluate_action_evidence(
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        learner_justification=justification,
        vlm_output=vlm_output_b,
        scenario_state=scenario_state,
    )
    elapsed_b = time.perf_counter() - started
    print(f"  verdict: {output_b['verdict']}  ({elapsed_b:.1f}s)")

    result = {
        "diagnostic": "vlm_security_relevance_dependency",
        "case": "access_key_info (identity-management, turn 1, expected weak support)",
        "expected_support_role": case["selected_evidence"]["expected_support_role"],
        "vlm_judgement_removed_fields": ["security_relevance", "supports_selected_action"],
        "diagnostic_a_production_input": {
            "vlm_output": vlm_output_a,
            "security_output": output_a,
            "elapsed_seconds": round(elapsed_a, 1),
        },
        "diagnostic_b_facts_only_input": {
            "vlm_output": vlm_output_b,
            "security_output": output_b,
            "elapsed_seconds": round(elapsed_b, 1),
        },
        "verdicts_match": output_a["verdict"] == output_b["verdict"],
    }

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved diagnostic result to {OUT_PATH}")
    print(f"Verdict A: {output_a['verdict']}  |  Verdict B: {output_b['verdict']}  |  match: {result['verdicts_match']}")


if __name__ == "__main__":
    main()
