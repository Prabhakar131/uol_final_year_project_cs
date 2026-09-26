from __future__ import annotations

from typing import Any


def to_camel_case_keys(value: dict[str, Any]) -> dict[str, Any]:
    """
    Converts a flat dict's snake_case keys to camelCase for the frontend JSON boundary.

    This only renames the dict's own keys - it does not recurse into nested
    dicts/lists, since callers pass through nested payloads (e.g. turn_config,
    evidence facts) whose internal snake_case keys the frontend already reads
    directly.
    """

    return {_snake_to_camel(key): item for key, item in value.items()}


def _snake_to_camel(key: str) -> str:
    if "_" not in key:
        return key

    first, *rest = key.split("_")
    return first + "".join(word.capitalize() for word in rest)
