"""Shape CloudWatch facts without inventing events, identities or correlations."""
from typing import Any

from cloudir.evidence_core.evidence_text_helpers import clean_value


def normalise_cloudwatch_facts(facts: dict[str, Any]) -> dict[str, Any]:
    def first(*keys, default="Unknown"):
        for key in keys:
            value = clean_value(facts.get(key))
            if value:
                return value
        return default

    # An explicitly empty log_rows must not fall through to another field.
    raw = next((facts[key] for key in ("log_rows", "rows", "events", "logs") if key in facts), [])
    if not isinstance(raw, list):
        raise ValueError("CloudWatch log_rows must be a list; regenerate the evidence.")
    rows = []
    for row in raw:
        if isinstance(row, list):
            if len(row) > 3:
                raise ValueError("CloudWatch rows require exactly three columns; extra content would be lost.")
            rows.append([clean_value(v) or "Unknown" for v in row] + ["Unknown"] * (3 - len(row)))
        elif isinstance(row, dict):
            def field(*keys):
                return next((clean_value(row[k]) for k in keys if k in row and clean_value(row[k])), "Unknown")
            rows.append([field("timestamp", "time", "event_time"),
                         field("log_stream", "stream", "logStream"),
                         field("message", "summary", "description")])
        else:
            raise ValueError("CloudWatch rows must be lists or objects; regenerate the evidence.")
    matched = first("matched_records", "matchedRecords")
    if matched.isdigit() and int(matched) < len(rows):
        raise ValueError("CloudWatch matched_records is smaller than its supplied row count.")
    # Preserve extra authored keys, but never turn evidence summaries or scenario
    # defaults into observed log messages. Unknowns remain visible unknowns.
    return {**facts, "log_rows": rows,
            "query": first("query"), "log_group": first("log_group", "logGroup"),
            "matched_records": matched, "scanned_bytes": first("scanned_bytes", "scannedBytes"),
            "time_range": first("time_range", "timeRange"),
            "alarm_state": first("alarm_state", "alarmState")}
