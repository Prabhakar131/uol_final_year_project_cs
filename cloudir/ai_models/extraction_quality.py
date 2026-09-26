"""Structural extraction checks, independent of evidence support labels.

Passing these checks does not establish transcription accuracy or a security
verdict. In particular, displayed rows need not equal the matched-record badge.
"""
from typing import Any
from datetime import datetime
import re


class ExtractionQualityError(ValueError):
    pass


def validate_extraction(output: dict[str, Any], evidence: dict[str, Any]) -> None:
    if not isinstance(output, dict):
        raise ExtractionQualityError("The extraction must be an object.")
    template = evidence.get("template") or evidence.get("type") or output.get("evidence_type")
    if template != "cloudwatch":
        facts = output.get("visible_facts_extracted")
        if not isinstance(facts, list) or not any(isinstance(f, str) and f.strip() for f in facts):
            raise ExtractionQualityError("No visible evidence facts were extracted.")
        return

    rows = output.get("event_rows")
    if not isinstance(rows, list):
        raise ExtractionQualityError("CloudWatch result rows were not extracted in the required format.")
    matched = output.get("matched_records")
    if matched is not None and (type(matched) is not int or matched < 0):
        raise ExtractionQualityError("The matched-record badge is invalid.")
    if not rows:
        # A readable zero-result query is evidence; missing rows with a positive
        # or unreadable count are an extraction failure, not weak learner work.
        if matched == 0 and all(isinstance(output.get(k), str) and output[k].strip()
                                for k in ("log_group", "time_range")):
            return
        raise ExtractionQualityError("No event rows were read; an empty results table was not established.")
    if matched is not None and len(rows) > matched:
        raise ExtractionQualityError("More rows were extracted than the matched-record badge reports.")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ExtractionQualityError(f"Event row {index} is malformed.")
        for key in ("timestamp", "log_stream", "message"):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise ExtractionQualityError(f"Event row {index} has an invalid {key}.")
        if type(row.get("truncated")) is not bool:
            raise ExtractionQualityError(f"Event row {index} has an invalid truncation flag.")
        timestamp = row.get("timestamp") or ""
        if re.match(r"^\d{4}-\d{2}-\d{2}T", timestamp):
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ExtractionQualityError(f"Event row {index} has a malformed ISO timestamp; re-read it without guessing missing digits.") from exc
    if not any(isinstance(row.get("message"), str) and row["message"].strip() for row in rows):
        raise ExtractionQualityError("Event messages are unreadable; headers alone cannot be judged.")


def security_observations(output: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    validate_extraction(output, evidence)
    if "event_rows" in output:
        # Rebuild from structured fields; discard free-form summaries and
        # duplicate compatibility facts that might turn editor text into proof.
        return {
            "event_rows": [{"row_number": i, **{k: row.get(k) for k in
                            ("timestamp", "log_stream", "message", "truncated")}}
                           for i, row in enumerate(output["event_rows"], 1)],
            "query_metadata_not_event_evidence": {k: output.get(k) for k in
                                                  ("query_text", "time_range", "log_group", "matched_records")},
            "extraction_warnings": output.get("extraction_warnings", []),
        }
    return {k: output[k] for k in ("evidence_type", "visible_facts_extracted",
                                   "important_visible_fields", "extraction_warnings") if k in output}
