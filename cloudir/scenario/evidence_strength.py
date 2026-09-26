"""Checks that each evidence item's visible content fits its support_role.

The security judge grades only what a screenshot shows, never the label. If a
"partial" or "weak" item shows as much incident activity as the "strong" one,
the judge correctly awards Strong Support and the learner is rewarded for the
wrong choice. These checks reject such turns at generation time so they are
regenerated instead of saved. They compare event names and wording; they are
a structural guard, not a proof that a label is semantically right.
"""
from __future__ import annotations

import re
from typing import Any

from cloudir.evidence_core.evidence_text_helpers import clean_value


# API calls that are security-relevant on their own: tampering with audit
# logging or detection, and creating or escalating credentials or permissions.
SENSITIVE_EVENTS = frozenset({
    "stoplogging", "deletetrail", "updatetrail", "puteventselectors",
    "deletedetector", "updatedetector", "deleteflowlogs",
    "createaccesskey", "createuser", "createloginprofile", "updateloginprofile",
    "attachuserpolicy", "putuserpolicy", "attachrolepolicy", "putrolepolicy",
    "createpolicyversion", "updateassumerolepolicy",
    "deactivatemfadevice", "deletemfadevice",
})

# Analyst conclusions written as if they were log output. Real consoles do not
# say this, and it hands the learner the answer the evidence should make them find.
CONCLUSION_PHRASES = ("same source ip", "observed across", "correlat", "linked to", "recommend", "confirm")

EVENT_TEMPLATES = frozenset({"cloudtrail", "iam_activity", "cloudwatch", "guardduty"})

# Generation prompts include this so the model knows the checks below.
SUPPORT_ROLE_CONTENT_RULES = """
EVIDENCE CONTENT RULES (turns that break these are rejected and regenerated):
- The grader sees only the rendered screenshot, never support_role. Label each item by what it visibly shows.
- strong: shows the event, finding or condition that directly justifies the best action.
- partial: relevant but incomplete. Do not repeat the strong item's main event or all of its events. Show at most one suspicious event (for example StopLogging, CreateAccessKey, or a ConsoleLogin without MFA). Leave the missing link out on purpose.
- weak: plausible cloud evidence with no suspicious events and none of the strong item's events, such as a status or inventory page, billing, or routine service logs.
- Rows, alarm states and risk flags record raw events only. Never write conclusions such as "same source IP observed across", "linked to", "correlated", "recommended" or "confirmed".
"""

_API_NAME = re.compile(
    r"\b((?:Create|Delete|Update|Put|Attach|Detach|List|Get|Describe|Stop|Start|Console|Assume"
    r"|Run|Terminate|Batch|Invoke|Enable|Disable|Deactivate|Remove|Add|Modify)[A-Z][A-Za-z]*)\b"
)
_NO_MFA = ("mfaauthenticated=false", "mfa false", "mfa=false", "without mfa")
_FAILED = ("fail", "denied")
_SERIOUS_SEVERITY = ("medium", "high", "critical")


def validate_support_role_content(evidence_facts: Any) -> None:
    problems = support_role_content_problems(evidence_facts)

    if problems:
        raise ValueError(
            "Evidence content does not match its support_role; regenerate the evidence "
            "content (changing labels is not a repair). " + " ".join(problems)
        )


def support_role_content_problems(evidence_facts: Any) -> list[str]:
    if not isinstance(evidence_facts, list):
        return []

    items = [item for item in evidence_facts if isinstance(item, dict)]
    strong = next((item for item in items if item.get("support_role") == "strong"), None)

    if strong is None:
        return []

    problems: list[str] = []
    strong_events = visible_events(strong)
    strong_suspicious = {name for name, suspicious in strong_events.items() if suspicious}
    strong_primary = primary_event(strong)

    if _template(strong) in EVENT_TEMPLATES and not strong_events:
        problems.append(f"Strong evidence {strong.get('id', 'unknown')} shows no event rows or finding.")

    for item in items:
        role = item.get("support_role")

        if role not in {"partial", "weak"}:
            continue

        label = f"{role.capitalize()} evidence {item.get('id', 'unknown')}"
        events = visible_events(item)
        suspicious = sorted(name for name, is_suspicious in events.items() if is_suspicious)
        phrase = _conclusion_phrase(item)

        if phrase:
            problems.append(f"{label} states a conclusion ('{phrase}') instead of recording an event.")

        if role == "partial":
            if strong_primary and strong_primary in events:
                problems.append(f"{label} shows {strong_primary}, the strong item's main event.")
            elif strong_events and set(strong_events) <= set(events):
                problems.append(f"{label} shows every event the strong item shows.")

            if strong_suspicious and len(suspicious) > 1:
                problems.append(
                    f"{label} shows {len(suspicious)} suspicious events ({', '.join(suspicious)}); "
                    "partial evidence may show at most one."
                )

        else:
            if suspicious:
                problems.append(f"{label} shows suspicious activity ({', '.join(suspicious)}); weak evidence must not.")

            shared = sorted(set(events) & set(strong_events))

            if shared:
                problems.append(f"{label} repeats the strong item's events ({', '.join(shared)}).")

    return problems


def visible_events(item: dict[str, Any]) -> dict[str, bool]:
    """Distinct event names an item shows, each mapped to whether it is suspicious."""

    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    template = _template(item)
    found: list[tuple[str, bool]] = []

    if template == "cloudtrail":
        if clean_value(facts.get("event_name")):
            context = f"mfa {clean_value(facts.get('mfa'))} {clean_value(facts.get('error_code'))}"
            found.append(_classify(clean_value(facts["event_name"]), context))

        for row in _rows(facts.get("related_events")):
            if len(row) > 1 and row[1]:
                found.append(_classify(row[1], " ".join(row)))

    elif template == "iam_activity":
        mfa_context = f"mfa {clean_value(facts.get('mfa'))}"

        for row in _rows(facts.get("activity_rows")):
            if len(row) > 2 and row[2]:
                found.append(_classify(row[2], f"{mfa_context} {' '.join(row)}"))

    elif template == "cloudwatch":
        for row in _rows(facts.get("log_rows")):
            message = row[2] if len(row) > 2 else " ".join(row)
            api_name = _API_NAME.search(message)
            found.append(_classify(api_name.group(1) if api_name else message, message))

    elif template == "guardduty":
        finding = clean_value(facts.get("finding_type"))

        if finding:
            found.append((finding.lower(), _is_serious_severity(facts.get("severity"))))

    events: dict[str, bool] = {}

    for name, suspicious in found:
        events[name] = events.get(name, False) or suspicious

    return events


def primary_event(item: dict[str, Any]) -> str:
    """The single headline event of a record view; timelines have none."""

    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    template = _template(item)

    if template == "cloudtrail":
        return clean_value(facts.get("event_name")).lower()

    if template == "guardduty":
        return clean_value(facts.get("finding_type")).lower()

    return ""


def event_is_suspicious(name: str, context: str = "") -> bool:
    """Whether an event name (with its row text as context) counts as suspicious here."""

    return _classify(name, context)[1]


def _classify(name: str, context: str) -> tuple[str, bool]:
    # "Create Access Key" and "CreateAccessKey" are the same event.
    key = re.sub(r"\s+", "", clean_value(name)).lower()
    context = context.lower()

    if key == "consolelogin":
        return key, any(marker in context for marker in _NO_MFA + _FAILED)

    return key, key in SENSITIVE_EVENTS


def _is_serious_severity(value: Any) -> bool:
    text = clean_value(value).lower()

    try:
        return float(text) >= 4
    except ValueError:
        return any(level in text for level in _SERIOUS_SEVERITY)


def _conclusion_phrase(item: dict[str, Any]) -> str:
    facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
    texts = [clean_value(facts.get("alarm_state"))]

    for key in ("related_events", "activity_rows", "log_rows"):
        texts.extend(" ".join(row) for row in _rows(facts.get(key)))

    if isinstance(facts.get("risk_flags"), list):
        texts.extend(clean_value(flag) for flag in facts["risk_flags"])

    combined = " ".join(texts).lower()

    return next((phrase for phrase in CONCLUSION_PHRASES if phrase in combined), "")


def _rows(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []

    rows = []

    for row in value:
        if isinstance(row, dict):
            row = list(row.values())

        if isinstance(row, list):
            rows.append([clean_value(cell) for cell in row])

    return rows


def _template(item: dict[str, Any]) -> str:
    return clean_value(item.get("template") or item.get("type")).lower()
