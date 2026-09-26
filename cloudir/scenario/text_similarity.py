from __future__ import annotations

import re
from difflib import SequenceMatcher

from cloudir.evidence_core.evidence_text_helpers import clean_value


def texts_match(left: str, right: str) -> bool:
    return (
        bool(left)
        and bool(right)
        and normalise_comparison_text(left) == normalise_comparison_text(right)
    )


def texts_too_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False

    left_normalised = normalise_comparison_text(left)
    right_normalised = normalise_comparison_text(right)

    if not left_normalised or not right_normalised:
        return False

    if left_normalised == right_normalised:
        return True

    return SequenceMatcher(None, left_normalised, right_normalised).ratio() >= 0.82


def normalise_comparison_text(value: str) -> str:
    value = clean_value(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def is_placeholder_text(value: str) -> bool:
    normalised = normalise_comparison_text(value)

    return normalised in {
        "",
        "unknown",
        "short phase name",
        "learner visible briefing for the next turn",
        "short learner facing guidance for choosing the next action",
    }


def join_sentences(prefix: str, original: str) -> str:
    prefix = clean_value(prefix)
    original = clean_value(original)

    if not prefix:
        return original

    if not original:
        return prefix

    if normalise_comparison_text(prefix) in normalise_comparison_text(original):
        return original

    return f"{prefix} {original}"
