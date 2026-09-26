from __future__ import annotations

import json
import re
from typing import Any


def clean_value(value: Any) -> str:
    """Coerces any AI-generated value (str/list/dict/None) into a plain string."""

    if value in [None, "", [], {}]:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        values = [clean_value(item) for item in value]
        return ", ".join(item for item in values if item)

    if isinstance(value, dict):
        for nested_key in [
            "value",
            "name",
            "id",
            "arn",
            "type",
            "summary",
            "description",
            "ip",
            "source_ip",
            "sourceIPAddress",
        ]:
            nested_value = value.get(nested_key)

            if nested_value not in [None, "", [], {}]:
                return clean_value(nested_value)

        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def build_evidence_text(evidence: dict[str, Any], facts: dict[str, Any]) -> str:
    metadata_text = " ".join(
        str(evidence.get(key, ""))
        for key in [
            "id",
            "title",
            "type",
            "template",
            "summary",
            "why_it_may_matter",
        ]
        if evidence.get(key)
    )

    facts_text = json.dumps(facts, ensure_ascii=False) if isinstance(facts, dict) else ""

    return f"{metadata_text} {facts_text}".strip()


def get_any(
    facts: dict[str, Any],
    evidence: dict[str, Any],
    *keys: str,
    default: str = "Unknown",
) -> str:
    for key in keys:
        value = clean_value(facts.get(key))

        if value and value.lower() != "unknown":
            return value

    for key in keys:
        value = clean_value(evidence.get(key))

        if value and value.lower() != "unknown":
            return value

    return default


def get_row_value(row: dict[str, Any], *keys: str, default: str = "Unknown") -> str:
    for key in keys:
        value = clean_value(row.get(key))

        if value and value.lower() != "unknown":
            return value

    return default


def find_ip(text: str, default: str = "Unknown") -> str:
    match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    return match.group(0) if match else default


def find_time(text: str, default: str = "Unknown") -> str:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\s?UTC\b",
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?Z\b",
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\b",
        r"\b\d{2}:\d{2}:\d{2}\s?UTC\b",
        r"\b\d{2}:\d{2}\s?UTC\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return default
