from __future__ import annotations

from typing import Any


SUPPORT_ROLE_SEQUENCE = ("strong", "partial", "weak")
SUPPORT_ROLE_SET = set(SUPPORT_ROLE_SEQUENCE)


def normalise_evidence_support_roles(evidence_facts: Any) -> None:
    """
    Normalises authored role spelling without guessing or reassigning labels.

    The field is internal scenario-design metadata. It should help generation
    produce one best item, one incomplete-but-relevant item, and one weaker
    distractor. Validation rejects invalid role sets instead of relabelling facts.
    """

    if not isinstance(evidence_facts, list):
        return

    for item in evidence_facts:
        if not isinstance(item, dict):
            continue

        # Normalise spelling only. Missing/duplicate roles must be corrected
        # during authoring, never assigned from position after content exists.
        item["support_role"] = normalise_support_role(item.get("support_role") or item.get("supportRole"))


def normalise_support_role(value: Any) -> str:
    role = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

    aliases = {
        "best": "strong",
        "strong_support": "strong",
        "direct": "strong",
        "partial_support": "partial",
        "reasonable": "partial",
        "incomplete": "partial",
        "weak_support": "weak",
        "distractor": "weak",
        "weaker": "weak",
    }

    role = aliases.get(role, role)

    return role if role in SUPPORT_ROLE_SET else ""
