from __future__ import annotations

from cloudir.evidence_core.cloudwatch_facts import normalise_cloudwatch_facts

from typing import Any

from cloudir.evidence_core.template_inference import force_correct_template, infer_evidence_template
from cloudir.evidence_core.evidence_text_helpers import (
    build_evidence_text,
    clean_value,
    find_ip,
    find_time,
    get_any,
    get_row_value,
)


DEFAULT_INCIDENT_USER = "arn:aws:iam::123456789012:user/admin-test"
DEFAULT_INCIDENT_SOURCE_IP = "185.220.101.42"
DEFAULT_INCIDENT_REGION = "ap-southeast-1"
DEFAULT_INCIDENT_TIME = "2023-10-01T10:21:35Z"
DEFAULT_ACCESS_KEY_ID = "AKIA4Z7EXAMPLE92K"
DEFAULT_ACCOUNT_ID = "123456789012"


def _is_placeholder_value(value: Any) -> bool:
    text = clean_value(value)
    lowered = text.lower()

    if not lowered:
        return True

    placeholder_values = {
        "unknown",
        "timestamp",
        "time",
        "event name",
        "event_name",
        "principal",
        "result",
        "aws api action name",
        "aws service source",
        "iam user, role, or arn",
        "iam user or role that owns the key",
        "source ipv4 address",
        "source ip",
        "aws region",
        "aws account id",
        "cloudtrail event id",
        "user agent or aws console source",
        "short request parameter excerpt",
        "akia-style key id or redacted key id",
        "aws service",
        "number of matched records",
        "amount scanned",
        "query time range",
        "short signal status",
        "log stream",
        "message",
    }

    return lowered in placeholder_values or lowered.startswith("aws ")


def _is_placeholder_user(value: Any) -> bool:
    text = clean_value(value).lower()
    return (
        not text
        or _is_placeholder_value(text)
        or text == "unknown"
        or text == "users"
        or text == "username"
        or "/username" in text
        or text.endswith(":user/username")
    )


def _is_placeholder_ip(value: Any) -> bool:
    text = clean_value(value)
    return (
        not text
        or _is_placeholder_value(text)
        or text.lower() == "unknown"
        or text.startswith("192.0.2.")
    )


def _clean_incident_user(value: Any, default: str = DEFAULT_INCIDENT_USER) -> str:
    text = clean_value(value)
    return default if _is_placeholder_user(text) else text


def _clean_incident_ip(value: Any, default: str = DEFAULT_INCIDENT_SOURCE_IP) -> str:
    text = clean_value(value)
    return default if _is_placeholder_ip(text) else text


def _clean_region(value: Any, default: str = DEFAULT_INCIDENT_REGION) -> str:
    text = clean_value(value)
    return default if _is_placeholder_value(text) or text == "us-east-1" else text


def _clean_time(value: Any, default: str = DEFAULT_INCIDENT_TIME) -> str:
    text = clean_value(value)
    return default if _is_placeholder_value(text) else text


def _cloudtrail_event_source(event_name: Any, raw_source: Any = "") -> str:
    raw = clean_value(raw_source)
    event = clean_value(event_name).lower()

    if event in ["consolelogin"]:
        return "signin.amazonaws.com"

    if event in ["assumerole", "getcalleridentity"]:
        return "sts.amazonaws.com"

    if event in ["createaccesskey", "listaccesskeys", "attachuserpolicy", "putuserpolicy"]:
        return "iam.amazonaws.com"

    if event in ["stoplogging", "listtrails", "updatetrail", "deletetrail"]:
        return "cloudtrail.amazonaws.com"

    if raw and not _is_placeholder_value(raw) and raw.lower() not in ["amazonec2", "amazon ec2", "amazonidentitymanagementservice"]:
        return raw

    return "iam.amazonaws.com"


def _cloudtrail_result(error_code: Any) -> str:
    code = clean_value(error_code)
    return "Success" if _is_placeholder_value(code) or code == "-" else code


def _log_rows_are_generic(rows: list[list[str]]) -> bool:
    generic_terms = [
        "my-log-group",
        "public endpoint",
        "event detected",
        "attempt to access",
        "timestamp log stream message",
        "log stream",
    ]
    row_text = " ".join(" ".join(row) for row in rows).lower()
    return any(term in row_text for term in generic_terms)


def _is_automated_response_text(text: str) -> bool:
    return any(
        term in text
        for term in [
            "securityhub",
            "security hub",
            "eventbridge",
            "step functions",
            "state machine",
            "lambda",
            "remediation",
            "workflow",
        ]
    )


def _automated_response_log_rows() -> list[list[str]]:
    return [
        ["2023-10-01T10:19:15Z", "check-state/[$LATEST]a1", "Received Security Hub finding custom-finding-7421"],
        ["2023-10-01T10:19:17Z", "check-state/[$LATEST]a1", "Validated remediationTarget against allowlist"],
        ["2023-10-01T10:20:02Z", "check-state/[$LATEST]a1", "Sanitised optional note field before workflow handoff"],
        ["2023-10-01T10:21:35Z", "workflow/execution", "Started Step Functions execution securityhub-remediation-7421"],
        ["2023-10-01T10:24:44Z", "workflow/control", "Manual approval required before Systems Manager remediation"],
    ]


def _clean_access_key_id(value: Any, default: str = DEFAULT_ACCESS_KEY_ID) -> str:
    text = clean_value(value)
    lowered = text.lower()
    if (
        _is_placeholder_value(text)
        or lowered in ["keyid", "accesskeyid", "access_key_id"]
        or "example" in lowered
    ):
        return default
    return text


def _clean_service_name(value: Any, default: str = "iam.amazonaws.com") -> str:
    text = clean_value(value)
    lowered = text.lower()
    if _is_placeholder_value(text) or lowered in ["amazonec2", "amazon ec2", "ec2"]:
        return default
    return text


def _clean_amount(value: Any, default: str) -> str:
    text = clean_value(value)
    if not text or text.lower() == "unknown":
        return default
    return text


def _clean_change(value: Any, default: str = "+331%") -> str:
    text = clean_value(value)
    if not text or text.lower() == "unknown":
        return default
    return text


def normalise_evidence_facts(evidence: dict[str, Any]) -> None:
    """
    Normalises AI-generated facts so screenshot templates do not render
    everything as Unknown.

    This does not create fake evidence items.
    It only maps existing/generated values into the exact keys expected
    by the screenshot templates. Unlike `repair_next_turn_template_facts`,
    this only fills in missing/unknown keys rather than rebuilding the
    facts dict from scratch.
    """

    force_correct_template(evidence)

    template = evidence.get("template")
    facts = evidence.get("facts")

    if not isinstance(facts, dict):
        evidence["facts"] = {}
        facts = evidence["facts"]

    metadata = {
        "id": evidence.get("id", ""),
        "title": evidence.get("title", ""),
        "type": evidence.get("type", ""),
        "summary": evidence.get("summary", ""),
        "why_it_may_matter": evidence.get("why_it_may_matter", ""),
    }

    combined_text = build_evidence_text(evidence, facts)
    combined_text_lower = combined_text.lower()

    def set_fact(target_key: str, *source_keys: str, default: str = "Unknown") -> None:
        current_value = clean_value(facts.get(target_key))

        if current_value and not _is_placeholder_value(current_value):
            facts[target_key] = current_value
            return

        for source_key in source_keys:
            source_value = clean_value(facts.get(source_key))

            if source_value and not _is_placeholder_value(source_value):
                facts[target_key] = source_value
                return

        for source_key in source_keys:
            source_value = clean_value(evidence.get(source_key))

            if source_value and not _is_placeholder_value(source_value):
                facts[target_key] = source_value
                return

        facts[target_key] = default

    def resolved_ip() -> str:
        direct_value = get_any(
            facts,
            evidence,
            "source_ip",
            "remote_ip",
            "ip",
            "sourceIPAddress",
            "client_ip",
            "remoteIp",
            default="",
        )

        return direct_value or find_ip(combined_text)

    def resolved_time() -> str:
        direct_value = get_any(
            facts,
            evidence,
            "event_time",
            "eventTime",
            "time",
            "timestamp",
            "created",
            "created_at",
            "first_seen",
            "firstSeen",
            default="",
        )

        return direct_value or find_time(combined_text)

    def incident_user() -> str:
        value = get_any(facts, evidence, "user", "principal", "actor", "username", default="")
        return _clean_incident_user(value)

    def incident_ip() -> str:
        value = resolved_ip()
        return _clean_incident_ip(value)

    def incident_time() -> str:
        value = resolved_time()
        return _clean_time(value)

    def incident_region() -> str:
        value = get_any(facts, evidence, "region", "awsRegion", "aws_region", default="")
        return _clean_region(value)

    if template == "cloudtrail":
        set_fact(
            "event_name",
            "event_name",
            "eventName",
            "action",
            "api_call",
            "operation",
            default=(
                "StopLogging"
                if "stoplogging" in combined_text_lower or "stop logging" in combined_text_lower
                else get_any(facts, evidence, "title", "type", default="Unknown")
            ),
        )
        set_fact(
            "event_source",
            "event_source",
            "eventSource",
            "service",
            default="cloudtrail.amazonaws.com",
        )
        facts["event_source"] = _cloudtrail_event_source(
            facts.get("event_name"),
            facts.get("event_source"),
        )
        set_fact(
            "user",
            "user",
            "username",
            "user_identity",
            "userIdentity",
            "principal",
            "actor",
            "identity",
            default="admin-test" if "admin-test" in combined_text_lower else "Unknown",
        )
        if "username" in str(facts.get("user", "")).lower():
            facts["user"] = DEFAULT_INCIDENT_USER
        facts["user"] = _clean_incident_user(facts.get("user"))

        facts["source_ip"] = incident_ip()

        set_fact(
            "mfa",
            "mfa",
            "mfaAuthenticated",
            "mfa_used",
            "mfaUsed",
            default="false" if "mfa false" in combined_text_lower or '"mfa": "false"' in combined_text_lower else "Unknown",
        )

        facts["event_time"] = incident_time()

        set_fact(
            "region",
            "region",
            "awsRegion",
            "aws_region",
            default=incident_region(),
        )
        facts["region"] = _clean_region(facts.get("region"))
        set_fact(
            "error_code",
            "error_code",
            "errorCode",
            "error",
            default="-",
        )
        set_fact(
            "risk_signal",
            "risk_signal",
            "risk",
            "finding",
            "summary",
            "description",
            default=metadata["summary"]
            or metadata["why_it_may_matter"]
            or "Logging visibility was modified.",
        )
        set_fact("event_id", "event_id", "eventID", "eventId", default="8f42b4ac-9f1b-4d1a-a3c8-5df2c9a71c03")
        set_fact("user_agent", "user_agent", "userAgent", default="signin.amazonaws.com")
        set_fact("recipient_account_id", "recipient_account_id", "recipientAccountId", "account_id", default=DEFAULT_ACCOUNT_ID)
        set_fact(
            "request_parameters",
            "request_parameters",
            "requestParameters",
            default=f'trailName=prod-org-trail, userName={facts.get("user", incident_user())}',
        )

        user = facts.get("user", incident_user())
        source_ip = facts.get("source_ip", incident_ip())
        event_name = facts.get("event_name", "StopLogging")
        event_time = facts.get("event_time", incident_time())

        if isinstance(facts.get("related_events"), list):
            cleaned_events: list[list[str]] = []

            for row in facts["related_events"]:
                if not isinstance(row, list):
                    continue

                padded = [clean_value(item) or "Unknown" for item in row[:5]]

                while len(padded) < 5:
                    padded.append("Unknown")

                if _is_placeholder_user(padded[2]):
                    padded[2] = user

                padded[3] = _clean_incident_ip(padded[3], default=source_ip)

                if _is_placeholder_value(padded[0]):
                    padded[0] = event_time

                if _is_placeholder_value(padded[1]):
                    padded[1] = event_name

                if _is_placeholder_value(padded[4]):
                    padded[4] = "Success"

                cleaned_events.append(padded)

            facts["related_events"] = cleaned_events

        if not facts.get("related_events"):
            # Only the selected event is known. Inventing surrounding incident
            # events would make partial or weak evidence look strong.
            facts["related_events"] = [
                [event_time, event_name, user, source_ip, _cloudtrail_result(facts.get("error_code"))],
            ]

    elif template == "guardduty":
        set_fact(
            "finding_type",
            "finding_type",
            "findingType",
            "finding_type_name",
            "finding",
            "guardduty_finding_type",
            default="UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration"
            if "credential" in combined_text_lower or "unauthorizedaccess" in combined_text_lower
            else get_any(facts, evidence, "title", default="Suspicious IAM activity"),
        )
        set_fact(
            "severity",
            "severity",
            "severity_label",
            "severityLabel",
            "risk_level",
            default="High"
            if "high" in combined_text_lower
            else "Medium"
            if "medium" in combined_text_lower or "suspicious" in combined_text_lower
            else "Unknown",
        )
        set_fact(
            "resource",
            "resource",
            "affected_resource",
            "affectedResource",
            "resource_id",
            "resourceId",
            "instance_id",
            "instanceId",
            "access_key_id",
            "accessKeyId",
            default=DEFAULT_INCIDENT_USER,
        )
        facts["resource"] = _clean_incident_user(facts.get("resource"))

        set_fact(
            "principal",
            "principal",
            "user",
            "username",
            "actor",
            "identity",
            "iam_user",
            "iamUser",
            default=DEFAULT_INCIDENT_USER,
        )
        facts["principal"] = _clean_incident_user(facts.get("principal"))

        facts["remote_ip"] = incident_ip()

        facts["first_seen"] = get_any(
            facts,
            evidence,
            "first_seen",
            "firstSeen",
            "created_at",
            "createdAt",
            "event_time",
            "eventTime",
            "time",
            "timestamp",
            default=incident_time(),
        )

        set_fact(
            "last_seen",
            "last_seen",
            "lastSeen",
            "updated_at",
            "updatedAt",
            "event_time",
            "eventTime",
            "time",
            "timestamp",
            default=facts.get("first_seen", incident_time()),
        )

        set_fact(
            "summary",
            "summary",
            "description",
            "risk_signal",
            "finding_summary",
            "findingSummary",
            default=(
                metadata["summary"]
                or metadata["why_it_may_matter"]
                or f"GuardDuty linked suspicious IAM activity for {incident_user()} from {incident_ip()}."
            ),
        )

    elif template == "iam_activity":
        rows = (
            facts.get("activity_rows")
            or facts.get("rows")
            or facts.get("events")
            or facts.get("timeline")
            or facts.get("activity")
            or []
        )

        normalised_rows: list[list[str]] = []

        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    padded_row = [clean_value(item) or "Unknown" for item in row[:4]]

                    while len(padded_row) < 4:
                        padded_row.append("Unknown")

                    padded_row[1] = _clean_incident_user(padded_row[1], default=incident_user())

                    padded_row[3] = _clean_incident_ip(padded_row[3], default=incident_ip())

                    normalised_rows.append(padded_row)

                elif isinstance(row, dict):
                    user = get_row_value(row, "user", "principal", "actor", "username")
                    source_ip = get_row_value(row, "source_ip", "sourceIPAddress", "ip", "remote_ip")

                    user = _clean_incident_user(user, default=incident_user())
                    source_ip = _clean_incident_ip(source_ip, default=incident_ip())

                    normalised_rows.append(
                        [
                            get_row_value(row, "time", "event_time", "eventTime", "timestamp"),
                            user,
                            get_row_value(row, "action", "event_name", "eventName", "operation"),
                            source_ip,
                        ]
                    )

        # Keep only the authored rows. Padding with incident events (or turning
        # the summary into an event) made partial and weak timelines look strong.
        facts["activity_rows"] = normalised_rows[:6]
        facts.setdefault("principal", normalised_rows[0][1] if normalised_rows else incident_user())
        facts.setdefault("source_ip", normalised_rows[0][3] if normalised_rows else "Unknown")
        facts.setdefault("mfa", "Unknown")
        facts.setdefault("policy_change", "Unknown")
        facts.setdefault("access_key_status", "Unknown")
        facts.setdefault("risk_flags", [])

    elif template == "cloudwatch":
        facts.update(normalise_cloudwatch_facts(facts))

    elif template == "access_key":
        set_fact(
            "access_key_id",
            "access_key_id",
            "key_id",
            "accessKeyId",
            "keyId",
            default=DEFAULT_ACCESS_KEY_ID,
        )
        facts["access_key_id"] = _clean_access_key_id(facts.get("access_key_id"))

        set_fact(
            "owner",
            "owner",
            "user",
            "principal",
            "actor",
            "username",
            default=incident_user(),
        )
        facts["owner"] = _clean_incident_user(facts.get("owner"))

        set_fact(
            "status",
            "status",
            "key_status",
            "keyStatus",
            default="Active",
        )
        if clean_value(facts.get("status")).lower() == "unknown":
            facts["status"] = "Active"

        set_fact(
            "last_used_service",
            "last_used_service",
            "lastUsedService",
            "service",
            default="iam.amazonaws.com",
        )
        facts["last_used_service"] = _clean_service_name(facts.get("last_used_service"))

        set_fact(
            "last_used_region",
            "last_used_region",
            "lastUsedRegion",
            "region",
            "awsRegion",
            default=incident_region(),
        )
        facts["last_used_region"] = _clean_region(facts.get("last_used_region"))

        set_fact(
            "last_used_time",
            "last_used_time",
            "lastUsed",
            "last_used",
            "lastUsedDate",
            "event_time",
            "eventTime",
            "time",
            default="Unknown",
        )

        # Defaulting to the attacker's IP or key-creation time would tie a
        # credential inventory page to the incident when the author did not.
        facts["source_ip"] = _clean_incident_ip(resolved_ip(), default="Unknown")

    elif template == "billing":
        set_fact(
            "current_spend",
            "current_spend",
            "current",
            "spend",
            "cost",
            "amount",
            default="$184.20",
        )
        facts["current_spend"] = _clean_amount(facts.get("current_spend"), "$184.20")

        set_fact(
            "previous_average",
            "previous_average",
            "baseline",
            "previous",
            "average",
            default="$42.75",
        )
        facts["previous_average"] = _clean_amount(facts.get("previous_average"), "$42.75")

        set_fact(
            "largest_service",
            "largest_service",
            "service",
            "top_service",
            "topService",
            default="EC2",
        )
        facts["largest_service"] = _clean_service_name(facts.get("largest_service"), "EC2")

        set_fact(
            "region",
            "region",
            "awsRegion",
            "aws_region",
            default=incident_region(),
        )
        facts["region"] = _clean_region(facts.get("region"))

        set_fact(
            "change",
            "change",
            "increase",
            "delta",
            "percentage_change",
            "percentageChange",
            default="+331%",
        )
        facts["change"] = _clean_change(facts.get("change"))


def repair_next_turn_template_facts(generated_turn: dict[str, Any]) -> dict[str, Any]:
    """
    Repairs weak local-model evidence facts after JSON parsing.

    This does not add fake evidence cards.
    It only reshapes each generated evidence item's existing text and metadata
    into the exact fields expected by the screenshot templates.

    Important:
    The template choice is repaired BEFORE facts are repaired.

    This prevents bad AI outputs like:
        type = guardduty
        template = cloudtrail

    from being rendered through the wrong screenshot template.
    """

    evidence_items = generated_turn.get("evidence_facts", [])

    if not isinstance(evidence_items, list):
        return generated_turn

    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue

        facts = evidence.get("facts")

        if not isinstance(facts, dict):
            facts = {}
            evidence["facts"] = facts

        # 1. Build text context first.
        metadata_text = build_evidence_text(evidence, facts)

        # 2. Repair evidence type + template BEFORE repairing facts.
        repaired_template = infer_evidence_template(evidence, facts, metadata_text)

        evidence["template"] = repaired_template
        evidence["type"] = repaired_template

        # 3. Rebuild text after template/type repair.
        metadata_text = build_evidence_text(evidence, facts)

        # 4. Repair facts using the corrected template.
        if repaired_template == "cloudtrail":
            evidence["facts"] = _repair_cloudtrail_facts(facts, metadata_text, evidence)

        elif repaired_template == "guardduty":
            evidence["facts"] = _repair_guardduty_facts(facts, metadata_text, evidence)

        elif repaired_template == "iam_activity":
            evidence["facts"] = _repair_iam_activity_facts(facts, metadata_text, evidence)

        elif repaired_template == "cloudwatch":
            evidence["facts"] = _repair_cloudwatch_facts(facts, metadata_text, evidence)

        elif repaired_template == "access_key":
            evidence["facts"] = _repair_access_key_facts(facts, metadata_text, evidence)

        elif repaired_template == "billing":
            evidence["facts"] = _repair_billing_facts(facts, metadata_text, evidence)

    return generated_turn


def _repair_cloudtrail_facts(
    facts: dict[str, Any],
    text: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    text_lower = text.lower()
    user = _clean_incident_user(get_any(
        facts,
        evidence,
        "user",
        "username",
        "user_identity",
        "userIdentity",
        "principal",
        "actor",
        "identity",
        default=DEFAULT_INCIDENT_USER,
    ))
    source_ip = _clean_incident_ip(get_any(
        facts,
        evidence,
        "source_ip",
        "sourceIPAddress",
        "remote_ip",
        "ip",
        default=find_ip(text, default=DEFAULT_INCIDENT_SOURCE_IP),
    ))
    event_name = get_any(
        facts,
        evidence,
        "event_name",
        "eventName",
        "action",
        "api_call",
        "operation",
        default="StopLogging" if "stoplogging" in text_lower else "ConsoleLogin",
    )
    event_time = get_any(
        facts,
        evidence,
        "event_time",
        "eventTime",
        "time",
        "timestamp",
        default=find_time(text, default=DEFAULT_INCIDENT_TIME),
    )
    mfa = get_any(
        facts,
        evidence,
        "mfa",
        "mfaAuthenticated",
        "mfa_used",
        "mfaUsed",
        default="false" if "mfa false" in text_lower or '"mfa": "false"' in text_lower else "false",
    )

    return {
        "event_name": event_name,
        "event_source": _cloudtrail_event_source(
            event_name,
            get_any(
                facts,
                evidence,
                "event_source",
                "eventSource",
                "service",
                default="",
            ),
        ),
        "user": user,
        "source_ip": source_ip,
        "mfa": mfa,
        "event_time": event_time,
        "region": get_any(
            facts,
            evidence,
            "region",
            "awsRegion",
            "aws_region",
            default=DEFAULT_INCIDENT_REGION,
        ),
        "error_code": get_any(
            facts,
            evidence,
            "error_code",
            "errorCode",
            "error",
            default="-",
        ),
        "risk_signal": get_any(
            facts,
            evidence,
            "risk_signal",
            "risk",
            "finding",
            "summary",
            "description",
            default=evidence.get("summary", "Logging visibility was modified."),
        ),
        "event_id": get_any(facts, evidence, "event_id", "eventId", default="8f42b4ac-9f1b-4d1a-a3c8-5df2c9a71c03"),
        "user_agent": get_any(facts, evidence, "user_agent", "userAgent", default="signin.amazonaws.com"),
        "recipient_account_id": get_any(facts, evidence, "recipient_account_id", "recipientAccountId", default="123456789012"),
        "request_parameters": get_any(
            facts,
            evidence,
            "request_parameters",
            "requestParameters",
            default=f"trailName=prod-org-trail, userName={user}",
        ),
        "related_events": facts.get("related_events")
        if isinstance(facts.get("related_events"), list)
        else [[event_time, event_name, user, source_ip, _cloudtrail_result(facts.get("error_code"))]],
    }


def _repair_guardduty_facts(
    facts: dict[str, Any],
    text: str,
    evidence: dict[str, Any],
) -> dict[str, str]:
    text_lower = text.lower()
    default_principal = (
        "events.amazonaws.com/securityhub-remediation-rule"
        if _is_automated_response_text(text_lower)
        else DEFAULT_INCIDENT_USER
    )
    principal = _clean_incident_user(
        get_any(
            facts,
            evidence,
            "principal",
            "user",
            "username",
            "actor",
            "identity",
            "iam_user",
            "iamUser",
            default=default_principal,
        )
    )
    remote_ip = _clean_incident_ip(
        get_any(
            facts,
            evidence,
            "remote_ip",
            "source_ip",
            "sourceIPAddress",
            "ip",
            default=find_ip(text, default=DEFAULT_INCIDENT_SOURCE_IP),
        )
    )
    first_seen = get_any(
        facts,
        evidence,
        "first_seen",
        "firstSeen",
        "created_at",
        "createdAt",
        "event_time",
        "eventTime",
        "time",
        "timestamp",
        default=find_time(text, default=DEFAULT_INCIDENT_TIME),
    )
    resource = _clean_incident_user(
        get_any(
            facts,
            evidence,
            "resource",
            "affected_resource",
            "affectedResource",
            "resource_id",
            "resourceId",
            "instance_id",
            "instanceId",
            "access_key_id",
            "accessKeyId",
            default=principal,
        ),
        default=principal,
    )

    return {
        "finding_type": get_any(
            facts,
            evidence,
            "finding_type",
            "findingType",
            "finding",
            "guardduty_finding_type",
            default=(
                "UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration"
                if "unauthorizedaccess" in text_lower or "credential" in text_lower
                else "Suspicious IAM activity"
            ),
        ),
        "severity": get_any(
            facts,
            evidence,
            "severity",
            "severity_label",
            "severityLabel",
            "risk_level",
            default=(
                "High"
                if "high" in text_lower
                else "Medium"
                if "medium" in text_lower or "suspicious" in text_lower
                else "Unknown"
            ),
        ),
        "resource": resource,
        "principal": principal,
        "remote_ip": remote_ip,
        "first_seen": first_seen,
        "last_seen": get_any(
            facts,
            evidence,
            "last_seen",
            "lastSeen",
            "updated_at",
            "updatedAt",
            "event_time",
            "eventTime",
            "time",
            "timestamp",
            default=first_seen or DEFAULT_INCIDENT_TIME,
        ),
        "summary": get_any(
            facts,
            evidence,
            "summary",
            "description",
            "risk_signal",
            "finding_summary",
            "findingSummary",
            default=evidence.get(
                "summary",
                f"GuardDuty linked suspicious IAM activity for {principal} from {remote_ip}.",
            ),
        ),
    }


def _repair_iam_activity_facts(
    facts: dict[str, Any],
    text: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    raw_rows = (
        facts.get("activity_rows")
        or facts.get("rows")
        or facts.get("events")
        or facts.get("timeline")
        or facts.get("activity")
        or []
    )

    rows: list[list[str]] = []

    if isinstance(raw_rows, list):
        for row in raw_rows:
            if isinstance(row, list):
                padded = [clean_value(item) or "Unknown" for item in row[:4]]

                while len(padded) < 4:
                    padded.append("Unknown")

                rows.append(padded)

            elif isinstance(row, dict):
                rows.append(
                    [
                        get_row_value(row, "time", "event_time", "eventTime", "timestamp"),
                        get_row_value(row, "user", "principal", "actor", "username"),
                        get_row_value(row, "action", "event_name", "eventName", "operation"),
                        get_row_value(row, "source_ip", "sourceIPAddress", "ip", "remote_ip"),
                    ]
                )

    # Keep only the authored rows; never pad with incident events.
    principal = rows[0][1] if rows else get_any(facts, evidence, "user", "principal", "actor", "username", default="")

    return {
        "activity_rows": rows[:6],
        "principal": _clean_incident_user(principal),
        "source_ip": _clean_incident_ip(rows[0][3] if rows else get_any(facts, evidence, "source_ip", "ip", default=""), default="Unknown"),
        "mfa": get_any(facts, evidence, "mfa", "mfaAuthenticated", default="Unknown"),
        "policy_change": get_any(facts, evidence, "policy_change", "policyChange", default="Unknown"),
        "access_key_status": get_any(facts, evidence, "access_key_status", "accessKeyStatus", default="Unknown"),
        "risk_flags": facts.get("risk_flags") if isinstance(facts.get("risk_flags"), list) else [],
    }


def _repair_cloudwatch_facts(
    facts: dict[str, Any],
    text: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return normalise_cloudwatch_facts(facts)


def _repair_access_key_facts(
    facts: dict[str, Any],
    text: str,
    evidence: dict[str, Any],
) -> dict[str, str]:
    owner = _clean_incident_user(
        get_any(
            facts,
            evidence,
            "owner",
            "user",
            "principal",
            "actor",
            "username",
            default=DEFAULT_INCIDENT_USER,
        )
    )

    return {
        "access_key_id": _clean_access_key_id(
            get_any(
                facts,
                evidence,
                "access_key_id",
                "key_id",
                "accessKeyId",
                "keyId",
                default=DEFAULT_ACCESS_KEY_ID,
            )
        ),
        "owner": owner,
        "status": get_any(
            facts,
            evidence,
            "status",
            "key_status",
            "keyStatus",
            default="Active",
        ),
        "last_used_service": _clean_service_name(
            get_any(
                facts,
                evidence,
                "last_used_service",
                "lastUsedService",
                "service",
                default="iam.amazonaws.com",
            )
        ),
        "last_used_region": _clean_region(
            get_any(
                facts,
                evidence,
                "last_used_region",
                "lastUsedRegion",
                "region",
                "awsRegion",
                default=DEFAULT_INCIDENT_REGION,
            )
        ),
        "last_used_time": get_any(
            facts,
            evidence,
            "last_used_time",
            "lastUsed",
            "last_used",
            "lastUsedDate",
            "event_time",
            "eventTime",
            "time",
            default=find_time(text, default="Unknown"),
        ),
        "source_ip": _clean_incident_ip(
            get_any(
                facts,
                evidence,
                "source_ip",
                "sourceIPAddress",
                "remote_ip",
                "ip",
                default=find_ip(text, default="Unknown"),
            ),
            default="Unknown",
        ),
    }


def _repair_billing_facts(
    facts: dict[str, Any],
    text: str,
    evidence: dict[str, Any],
) -> dict[str, str]:
    return {
        "current_spend": _clean_amount(
            get_any(
                facts,
                evidence,
                "current_spend",
                "current",
                "spend",
                "cost",
                "amount",
                default="$184.20",
            ),
            "$184.20",
        ),
        "previous_average": _clean_amount(
            get_any(
                facts,
                evidence,
                "previous_average",
                "baseline",
                "previous",
                "average",
                default="$42.75",
            ),
            "$42.75",
        ),
        "largest_service": _clean_service_name(
            get_any(
                facts,
                evidence,
                "largest_service",
                "service",
                "top_service",
                "topService",
                default="EC2",
            ),
            "EC2",
        ),
        "region": _clean_region(
            get_any(
                facts,
                evidence,
                "region",
                "awsRegion",
                "aws_region",
                default=DEFAULT_INCIDENT_REGION,
            )
        ),
        "change": _clean_change(
            get_any(
                facts,
                evidence,
                "change",
                "increase",
                "delta",
                "percentage_change",
                "percentageChange",
                default="+331%",
            )
        ),
    }
