"""Read-only model diagnostic: no application state or saved runs are modified."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cloudir.ai_models.security_model import evaluate_action_evidence, unload_security_model, _device, security_model_label


def main():
    source = ROOT / "notebooks/model_benchmarks/results/experiments/cloudwatch_stage1_extraction.json"
    observed = json.loads(source.read_text())["output"]
    action = {"id": "review_iam_activities", "title": "Review Recent IAM Activities",
              "description": "Look at the latest IAM activities to identify potential threats."}
    evidence = {"id": "cloudwatch_insights_query", "template": "cloudwatch", "type": "cloudwatch",
                "title": "CloudWatch Log Insight Query"}
    justification = {"transcript": "I selected Review Recent IAM Activities to investigate this incident. I chose this screenshot because it might help review the recent activity. I will check the visible identity, source IP, event time, service, status, and risk signal to see what the evidence actually establishes. If those details do not prove the action, I would compare this with another source before drawing a conclusion."}
    unrelated = copy.deepcopy(observed)
    unrelated["event_rows"] = [{"timestamp": "2023-10-01T10:12:10Z", "log_stream": "app/health",
                                "message": "Public status endpoint returned HTTP 200; no identity activity recorded in this row.",
                                "truncated": False}]
    unrelated["matched_records"] = 1
    unrelated["extraction_warnings"] = []
    # Deliberately keep the old summaries and provocative editor description:
    # production must discard summaries and treat editor text only as context.
    cases = [("verified_saved_cloudwatch", observed), ("controlled_unrelated_event_same_query", unrelated),
             ("missing_rows_must_block", {**observed, "event_rows": []})]
    report = {"model_id": os.getenv("HF_SECURITY_MODEL_ID"), "security_weights": security_model_label(), "device": _device(),
              "security_source_sha256": hashlib.sha256((ROOT / "cloudir/ai_models/security_model.py").read_bytes()).hexdigest(),
              "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "scope": "Security-only diagnostic; controlled second/third inputs are not screenshot extractions.", "cases": []}
    # Exclusive creation prevents overwriting an earlier diagnostic.
    output = ROOT / "notebooks/model_benchmarks/results/experiments" / f"stage2_security_check_{time.time_ns()}.json"
    try:
        for name, facts in cases:
            print(f"Running {name} on {_device()}", flush=True)
            start = time.monotonic()
            record = {"name": name, "input": facts}
            try:
                record["output"] = evaluate_action_evidence(action, evidence, justification, facts, {})
            except Exception as exc:
                record["error"] = str(exc)
                record["error_type"] = type(exc).__name__
            record["elapsed_seconds"] = time.monotonic() - start
            report["cases"].append(record)
            print(json.dumps(record, indent=2), flush=True)
    finally:
        unload_security_model()
        with output.open("x") as file:
            json.dump(report, file, indent=2)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
