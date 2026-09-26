"""Live security/coach check; no app progress or saved-run mutation."""
from __future__ import annotations

import copy
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cloudir.ai_models import security_model, coach_model
from cloudir.ai_models.learner_grounding import learner_statements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral-only", action="store_true")
    args = parser.parse_args()
    source = ROOT / "notebooks/model_benchmarks/results/experiments/stage2_security_check_1789897549586437000.json"
    facts = json.loads(source.read_text())["cases"][0]["input"]
    action = {"id": "review_iam_activities", "title": "Review Recent IAM Activities",
              "description": "Look at the latest IAM activities to identify potential threats."}
    evidence = {"id": "cloudwatch_insights_query", "template": "cloudwatch", "type": "cloudwatch",
                "title": "CloudWatch Log Insight Query"}
    neutral = "I selected Review Recent IAM Activities to investigate this incident. I chose this screenshot because it might help review the recent activity. I will check the visible identity, source IP, event time, service, status, and risk signal to see what the evidence actually establishes. If those details do not prove the action, I would compare this with another source before drawing a conclusion."
    cases = [("original_neutral_transcript", neutral),
             ("explicit_truncation_statement", "Rows 1 and 4 have truncated messages. I cannot infer the hidden characters."),
             ("empty_transcript_with_leading_prompt", "")]
    if args.neutral_only:
        cases = cases[:1]
    folder = ROOT / "notebooks/model_benchmarks/results/experiments"
    output = folder / f"learner_grounding_{time.time_ns()}.json"
    report = {"source": str(source.relative_to(ROOT)), "device": security_model._device(),
              "scope": "Live security and coach on verified saved extraction; no app state updates.", "cases": []}
    try:
        for name, transcript in cases:
            justification = {"transcript": transcript, "prompt": "Explain the truncated messages and acknowledge the missing text."}
            record = {"name": name, "justification": justification}
            raw_responses = []
            original_generate = security_model._generate_security_response
            def capture_response(**kwargs):
                raw = original_generate(**kwargs)
                raw_responses.append(raw)
                print(raw, flush=True)
                return raw
            security_model._generate_security_response = capture_response
            start = time.monotonic()
            print(f"Security: {name}", flush=True)
            try:
                result = security_model.evaluate_action_evidence(action, evidence, justification, copy.deepcopy(facts), {})
                record["security"] = result
                statements = {item["statement_id"]: item["text"] for item in learner_statements(justification)}
                record["checks"] = {
                    "quotes_match_complete_statements": all(item["transcript_quote"] == statements[item["statement_id"]]
                                                            for item in result["justification_claims"]),
                    "no_truncation_credit_when_unsaid": name == "explicit_truncation_statement" or
                                                       "truncat" not in result["justification_assessment"].lower(),
                    "empty_has_no_claims": bool(transcript) or result["justification_claims"] == [],
                }
            except Exception as exc:
                record["security_error"] = str(exc)
                record["security_error_cause"] = str(exc.__cause__)
            finally:
                security_model._generate_security_response = original_generate
                record["raw_security_responses"] = raw_responses
            record["security_seconds"] = time.monotonic() - start
            report["cases"].append(record)
            print(json.dumps(record, indent=2), flush=True)
        security_model.unload_security_model()
        for record in report["cases"]:
            if "security" not in record:
                continue
            print(f"Coach: {record['name']}", flush=True)
            start = time.monotonic()
            try:
                record["coach"] = coach_model.generate_coach_feedback(
                    action, evidence, record["justification"], facts, record["security"], {})
            except Exception as exc:
                record["coach_error"] = str(exc)
                record["coach_error_cause"] = str(exc.__cause__)
            record["coach_seconds"] = time.monotonic() - start
            print(json.dumps(record, indent=2), flush=True)
    finally:
        security_model.unload_security_model()
        coach_model.unload_coach_model()
        with output.open("x") as file:
            json.dump(report, file, indent=2)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
