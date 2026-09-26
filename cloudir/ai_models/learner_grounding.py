"""Keep learner attribution extractive rather than model-written paraphrases."""
from __future__ import annotations

import re
from typing import Any


class LearnerGroundingError(ValueError):
    pass


ASSESSMENTS = {
    "supported": "This statement is consistent with the available evidence; it does not establish anything beyond the quoted statement.",
    "overclaim": "This statement claims more than the available evidence establishes.",
    "needs_detail": "This statement needs a more specific connection between a visible fact and the selected action.",
}


def learner_statements(justification: dict[str, Any] | None) -> list[dict[str, Any]]:
    data = justification or {}
    # An explicitly empty canonical transcript must not fall back to stale text.
    text = next((data[key] for key in ("transcript", "text", "typed_text", "typedText")
                 if key in data), "")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise LearnerGroundingError("Learner transcript must be text.")
    # Full statements keep negation and qualifications; the model cannot select
    # only 'acknowledge truncation' from 'I did not acknowledge truncation'.
    return [{"statement_id": i, "text": sentence}
            for i, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text.strip()), 1)
            if sentence]


def ground_justification(claims: Any, statements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(claims, list) or len(claims) > 3:
        raise LearnerGroundingError("justification_claims must be a list of at most three statement assessments.")
    by_id = {item["statement_id"]: item["text"] for item in statements}
    grounded = []
    used = set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"statement_id", "assessment"}:
            raise LearnerGroundingError("Each learner assessment needs only statement_id and assessment; no invented paraphrases.")
        ref, assessment = claim["statement_id"], claim["assessment"]
        if type(ref) is not int or ref not in by_id or ref in used:
            raise LearnerGroundingError("Learner assessment references a missing or duplicate transcript statement.")
        if not isinstance(assessment, str) or assessment not in ASSESSMENTS:
            raise LearnerGroundingError("Invalid learner statement assessment.")
        used.add(ref)
        grounded.append({**claim, "transcript_quote": by_id[ref]})
    if not statements:
        return [], "No learner justification was provided. The evidence verdict does not imply demonstrated learner reasoning."
    if not grounded:
        return [], "No statement-level assessment was established. No specific reasoning is credited to the learner."
    rendered = "\n\n".join(f'Learner statement: “{item["transcript_quote"]}”\n{ASSESSMENTS[item["assessment"]]}'
                           for item in grounded)
    return grounded, rendered


def validate_evidence_only_prose(result: dict[str, Any], fields: tuple[str, ...]) -> None:
    """Conservative tripwire, not a general semantic-entailment classifier.

    Personal assessment belongs exclusively to server-rendered statement quotes.
    Model prose explains evidence and future checks, not learner understanding.
    """
    attribution = re.compile(
        r"\b(learner|student|trainee|you|your|yours|yourself|transcript|justification)\b"
        r"|\b(great work|good job|well done|good instinct|right to notice)\b"
        r"|\b(correctly|rightly|accurately|appropriately)\s+(identified|noted|noticed|recogn[iz]ed|understood|explained|mentioned|acknowledged)\b",
        re.IGNORECASE,
    )
    for field in fields:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise LearnerGroundingError(f"Missing readable {field}.")
        if attribution.search(value):
            raise LearnerGroundingError(f"{field} contains learner attribution; use impersonal evidence analysis, not praise or a transcript paraphrase.")


def keep_evidence_only_prose(result: dict[str, Any], fields: tuple[str, ...]) -> dict[str, list[str]]:
    """Drop personal-ascription sentences, never rewrite them as grounded praise.

    Future suggestions/risks can be made impersonal mechanically. If filtering
    leaves a required field empty, validation still fails and triggers a retry.
    The audit records affected fields, not the discarded ungrounded claims.
    """
    removed, neutralised = [], []
    for field in fields:
        value = result.get(field)
        if not isinstance(value, str):
            continue
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", value.strip()):
            neutral = re.sub(r"^(?:the learner|you)\s+should\s+", "", sentence, flags=re.IGNORECASE)
            neutral = re.sub(r"^(?:the learner|you)\s+(?:might|may|could)\s+",
                             "A possible mistake is to ", neutral, flags=re.IGNORECASE)
            if neutral != sentence:
                neutral = neutral[:1].upper() + neutral[1:]
                neutralised.append(field)
            try:
                validate_evidence_only_prose({field: neutral}, (field,))
            except LearnerGroundingError:
                removed.append(field)
                continue
            kept.append(neutral)
        result[field] = " ".join(kept)
    validate_evidence_only_prose(result, fields)
    return {"removed_attribution_fields": sorted(set(removed)),
            "neutralised_advice_fields": sorted(set(neutralised))}
