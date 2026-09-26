from __future__ import annotations

from typing import Any

from cloudir.evidence_core.schemas import ALLOWED_TEMPLATES, TYPE_TO_TEMPLATE
from cloudir.evidence_core.evidence_text_helpers import build_evidence_text, clean_value, contains_any

TEMPLATE_NAME_ALIASES = {
    "guard duty": "guardduty",
    "guard_duty": "guardduty",
    "iam activity": "iam_activity",
    "iam-activity": "iam_activity",
    "iam timeline": "iam_activity",
    "iam user activity": "iam_activity",
    "iam_user_activity": "iam_activity",
    "cloud watch": "cloudwatch",
    "cloud-watch": "cloudwatch",
    "access key": "access_key",
    "access-key": "access_key",
    "accesskey": "access_key",
    "cost": "billing",
    "costs": "billing",
    "spend": "billing",
    "cloud trail": "cloudtrail",
    "cloud-trail": "cloudtrail",
}


def normalise_template_name(value: Any) -> str:
    raw = clean_value(value).lower().strip()
    return TEMPLATE_NAME_ALIASES.get(raw, raw)


def infer_evidence_template(
    evidence: dict[str, Any],
    facts: dict[str, Any] | None = None,
    text: str | None = None,
) -> str:
    """
    Infers the correct evidence template from id/title/type/template/summary/facts.

    Priority:
    1. Strong structural fact keys such as log_rows, activity_rows, finding_type.
    2. Strong identity words in id/title, such as CloudWatch or GuardDuty.
    3. Existing valid matching type/template pair.
    4. Existing valid type.
    5. Existing valid template.
    6. Broad keyword inference from full evidence text.
    7. CloudTrail fallback.

    This prevents CloudWatch log evidence that mentions IAM activity from being
    incorrectly converted into the IAM Activity screenshot template.
    """

    if facts is None:
        raw_facts = evidence.get("facts")
        facts = raw_facts if isinstance(raw_facts, dict) else {}

    if text is None:
        text = build_evidence_text(evidence, facts)

    text_lower = text.lower()

    existing_type = normalise_template_name(evidence.get("type"))
    existing_template = normalise_template_name(evidence.get("template"))

    evidence_id = clean_value(evidence.get("id")).lower()
    evidence_title = clean_value(evidence.get("title")).lower()
    evidence_identity_text = f"{evidence_id} {evidence_title}".strip()

    # 1. Strong structural fact keys are the safest signal.
    # These should beat broad keywords inside log messages.
    if "log_rows" in facts:
        return "cloudwatch"

    if "activity_rows" in facts:
        return "iam_activity"

    if any(
        key in facts
        for key in [
            "finding_type",
            "findingType",
            "guardduty_finding_type",
            "severity",
        ]
    ):
        return "guardduty"

    if any(
        key in facts
        for key in [
            "access_key_id",
            "accessKeyId",
            "last_used_service",
            "lastUsedService",
            "key_status",
            "keyStatus",
        ]
    ):
        return "access_key"

    if any(
        key in facts
        for key in [
            "current_spend",
            "previous_average",
            "largest_service",
            "cost",
            "spend",
        ]
    ):
        return "billing"

    if any(
        key in facts
        for key in [
            "event_name",
            "eventName",
            "event_source",
            "eventSource",
            "event_time",
            "eventTime",
        ]
    ):
        return "cloudtrail"

    # 2. Strong id/title identity should beat noisy evidence content.
    # Example: title="CloudWatch Monitoring Logs" should stay CloudWatch,
    # even if the log message mentions IAM, ConsoleLogin, or CreateAccessKey.
    if contains_any(
        evidence_identity_text,
        [
            "cloudwatch",
            "cloud watch",
            "cloudwatch logs",
            "monitoring logs",
            "log events",
            "logs insights",
        ],
    ):
        return "cloudwatch"

    if contains_any(
        evidence_identity_text,
        [
            "guardduty",
            "guard duty",
            "guardduty alert",
            "guardduty finding",
            "threat detection",
        ],
    ):
        return "guardduty"

    if contains_any(
        evidence_identity_text,
        [
            "iam activity",
            "iam_activity",
            "activity timeline",
            "user activity timeline",
            "iam timeline",
        ],
    ):
        return "iam_activity"

    if contains_any(
        evidence_identity_text,
        [
            "access key",
            "access_key",
            "access keys",
            "accesskey",
            "credential usage",
        ],
    ):
        return "access_key"

    if contains_any(
        evidence_identity_text,
        [
            "billing",
            "cost explorer",
            "usage charges",
        ],
    ):
        return "billing"

    if contains_any(
        evidence_identity_text,
        [
            "cloudtrail",
            "cloud trail",
            "event history",
            "api activity",
            "api call",
        ],
    ):
        return "cloudtrail"

    # 3. If type/template are already valid and aligned, trust them.
    if (
        existing_type in TYPE_TO_TEMPLATE
        and existing_template in ALLOWED_TEMPLATES
        and TYPE_TO_TEMPLATE[existing_type] == existing_template
    ):
        return existing_template

    # 4. Trust valid type before template because type is usually semantic.
    if existing_type in TYPE_TO_TEMPLATE:
        return TYPE_TO_TEMPLATE[existing_type]

    # 5. Trust valid template if type is missing or invalid.
    if existing_template in ALLOWED_TEMPLATES:
        return existing_template

    # 6. Broad keyword inference only after safer checks above.
    if contains_any(
        text_lower,
        [
            "cloudwatch",
            "cloud watch",
            "log group",
            "log stream",
            "log_rows",
            "cloudwatch logs",
            "metric alarm",
            "logs insights",
        ],
    ):
        return "cloudwatch"

    if contains_any(
        text_lower,
        [
            "guardduty",
            "guard duty",
            "finding_type",
            "findingtype",
            "unauthorizedaccess",
            "recon:iamuser",
            "stealth:iamuser",
            "credentialexfiltration",
            "anomalousbehavior",
            "suspicious guardduty",
        ],
    ):
        return "guardduty"

    if contains_any(
        text_lower,
        [
            "iam_activity",
            "iam activity",
            "activity timeline",
            "user activity timeline",
            "activity_rows",
            "consolelogin",
            "attachuserpolicy",
            "putuserpolicy",
            "listaccesskeys",
        ],
    ):
        return "iam_activity"

    if contains_any(
        text_lower,
        [
            "access_key",
            "access key",
            "accesskey",
            "access_key_id",
            "accesskeyid",
            "last_used_service",
            "lastusedservice",
            "key status",
            "key_status",
            "createaccesskey",
        ],
    ):
        return "access_key"

    if contains_any(
        text_lower,
        [
            "billing",
            "cost",
            "spend",
            "current_spend",
            "previous_average",
            "largest_service",
            "aws cost",
            "cost explorer",
        ],
    ):
        return "billing"

    if contains_any(
        text_lower,
        [
            "cloudtrail",
            "cloud trail",
            "stoplogging",
            "startlogging",
            "event_name",
            "eventname",
            "event_source",
            "eventsource",
            "sourceipaddress",
            "source_ip",
            "mfaauthenticated",
            "awsregion",
        ],
    ):
        return "cloudtrail"

    return "cloudtrail"


def force_correct_template(evidence: dict[str, Any]) -> None:
    """
    Corrects wrong AI-generated template/type combinations.

    Important:
    If the AI already generated a valid matching type/template pair,
    trust it first instead of trying to infer from title or facts.

    This prevents evidence such as CloudWatch logs that mention IAM activity
    from being incorrectly converted into the IAM Activity template.
    """

    existing_template = normalise_template_name(evidence.get("template"))
    existing_type = normalise_template_name(evidence.get("type"))

    if (
        existing_template in ALLOWED_TEMPLATES
        and existing_type in TYPE_TO_TEMPLATE
        and TYPE_TO_TEMPLATE[existing_type] == existing_template
    ):
        evidence["template"] = existing_template
        evidence["type"] = existing_template
        return

    if existing_type in TYPE_TO_TEMPLATE:
        evidence["template"] = TYPE_TO_TEMPLATE[existing_type]
        evidence["type"] = TYPE_TO_TEMPLATE[existing_type]
        return

    if existing_template in ALLOWED_TEMPLATES:
        evidence["template"] = existing_template
        evidence["type"] = existing_template
        return

    inferred_template = infer_evidence_template(evidence)
    evidence["template"] = inferred_template
    evidence["type"] = inferred_template


def validate_template_alignment(evidence: dict[str, Any]) -> None:
    evidence_id = evidence.get("id", "unknown")
    evidence_type = evidence.get("type")
    template = evidence.get("template")

    if template not in ALLOWED_TEMPLATES:
        raise ValueError(
            f"Evidence item {evidence_id} has invalid template={template}."
        )

    expected_template = TYPE_TO_TEMPLATE.get(evidence_type)

    if expected_template and expected_template != template:
        raise ValueError(
            f"Evidence item {evidence_id} has type={evidence_type} "
            f"but template={template}. Expected template={expected_template}."
        )
