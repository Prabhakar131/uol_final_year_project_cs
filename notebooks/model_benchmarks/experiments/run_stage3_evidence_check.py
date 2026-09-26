"""Render and read new CloudWatch screenshots without touching active runs."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cloudir.evidence_core.fact_repair import normalise_evidence_facts
from cloudir.evidence.templates.cloudwatch_template import render_cloudwatch
from cloudir.ai_models.image_model import analyse_evidence_image, unload_image_model, _device


def main():
    source = ROOT / "data/runs/run-906029b52954449c852ee52deef09c37/checkpoints/cp-b89ad40698004c9fa28c6d7f70c554bf/runtime/turns/turn_1/evidence_facts.json"
    original = next(e for e in json.loads(source.read_text()) if e["id"] == "cloudwatch_insights_query")
    weak = {"id": "cloudwatch_health_check", "title": "Service health check", "type": "cloudwatch",
            "template": "cloudwatch", "support_role": "weak", "summary": "Unrelated service status",
            "facts": {"log_rows": [["2023-10-01T10:12:10Z", "app/health", "Public endpoint returned HTTP 200"]],
                      "query": "fields @timestamp, @logStream, @message | filter @logStream = 'app/health'",
                      "log_group": "/app/health", "matched_records": "1", "scanned_bytes": "1 KB",
                      "time_range": "2023-10-01 10:10-10:30 UTC", "alarm_state": "OK"}}
    folder = ROOT / "notebooks/model_benchmarks/results/experiments" / f"stage3_evidence_{time.time_ns()}"
    folder.mkdir()
    report = {"model_id": os.getenv("HF_IMAGE_MODEL_ID"), "device": _device(),
              "scope": "Rendered previews and VLM only; no security, coach or app state updates.",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": []}
    try:
        for name, item in (("saved_five_rows", original), ("weak_health_check", weak)):
            evidence = copy.deepcopy(item)
            normalise_evidence_facts(evidence)
            image = folder / f"{name}.png"
            render_cloudwatch(evidence, image)
            record = {"name": name, "image": str(image.relative_to(ROOT)),
                      "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                      "facts_preserved": evidence["facts"]["log_rows"] == item["facts"]["log_rows"]}
            print(f"Extracting {name} on {_device()}", flush=True)
            start = time.monotonic()
            try:
                # Only template identity and pixels are given to the VLM.
                result = analyse_evidence_image(image, {}, {"template": "cloudwatch"})
                record["output"] = result
                expected = evidence["facts"]["log_rows"]
                rows = result.get("event_rows", [])
                def compact(value):
                    return re.sub(r"\s+", " ", str(value or "")).strip()
                record["checks"] = {
                    "row_count": len(rows) == len(expected),
                    "row_fields": len(rows) == len(expected) and all(
                        compact(row.get(key)) == compact(value)
                        for row, values in zip(rows, expected)
                        for key, value in zip(("timestamp", "log_stream", "message"), values)),
                    "no_truncation": bool(rows) and all(row.get("truncated") is False for row in rows),
                    "time_range": result.get("time_range") == evidence["facts"]["time_range"],
                }
            except Exception as exc:
                record["error"] = str(exc)
            record["elapsed_seconds"] = time.monotonic() - start
            report["cases"].append(record)
            print(json.dumps(record, indent=2), flush=True)
    finally:
        unload_image_model()
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        print(f"Saved {folder / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
