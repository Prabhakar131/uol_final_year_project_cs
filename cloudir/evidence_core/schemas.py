from __future__ import annotations


EVIDENCE_TEMPLATE_SCHEMAS = """
STRICT TEMPLATE FACT SCHEMAS:

If template is "cloudtrail", facts MUST contain:
{
  "event_name": "AWS API action name",
  "event_source": "AWS service source",
  "user": "IAM user, role, or ARN",
  "source_ip": "source IPv4 address",
  "mfa": "true, false, or unknown",
  "event_time": "timestamp",
  "region": "AWS region",
  "error_code": "error code or -",
  "risk_signal": "short risk explanation",
  "event_id": "CloudTrail event id",
  "user_agent": "user agent or AWS console source",
  "recipient_account_id": "AWS account id",
  "request_parameters": "short request parameter excerpt",
  "related_events": [
    ["time", "event name", "principal", "source ip", "result"]
  ]
}

If template is "guardduty", facts MUST contain:
{
  "finding_type": "GuardDuty finding type",
  "severity": "Low, Medium, High, or numeric severity",
  "resource": "affected IAM user, instance, role, or access key",
  "principal": "IAM user, role, or principal involved",
  "remote_ip": "remote IPv4 address",
  "first_seen": "timestamp",
  "last_seen": "timestamp",
  "summary": "short finding summary"
}

If template is "iam_activity", facts MUST contain:
{
  "principal": "selected IAM user, role, or principal",
  "source_ip": "source IPv4 address",
  "mfa": "true, false, or unknown",
  "policy_change": "short policy or permission context",
  "access_key_status": "short credential status context",
  "risk_flags": ["short risk flag", "short risk flag"],
  "activity_rows": [
    ["time", "user", "action", "source ip"],
    ["time", "user", "action", "source ip"]
  ]
}

If template is "cloudwatch", facts MUST contain:
{
  "query": "CloudWatch Logs Insights query",
  "log_group": "AWS log group name",
  "matched_records": "number of matched records",
  "scanned_bytes": "amount scanned",
  "time_range": "query time range",
  "alarm_state": "short signal status",
  "log_rows": [
    ["timestamp", "log stream", "message"],
    ["timestamp", "log stream", "message"]
  ]
}

If template is "access_key", facts MUST contain:
{
  "access_key_id": "AKIA-style key id or redacted key id",
  "owner": "IAM user or role that owns the key",
  "status": "Active or Inactive",
  "last_used_service": "AWS service",
  "last_used_region": "AWS region",
  "last_used_time": "timestamp",
  "source_ip": "source IPv4 address"
}

If template is "billing", facts MUST contain:
{
  "current_spend": "current spend amount",
  "previous_average": "previous average spend amount",
  "largest_service": "largest AWS service by cost",
  "region": "AWS region",
  "change": "percentage or amount increase"
}

Template rules:
- Each evidence item MUST use one of the strict facts schemas above.
- Do not use invented fact keys such as guardduty_findings, findings, details, data, records, events_summary, evidence, or notes.
- Do not leave facts as an empty object.
- Do not put all facts into a single summary string.
"""

ALLOWED_TEMPLATES = {
    "cloudtrail",
    "iam_activity",
    "guardduty",
    "cloudwatch",
    "access_key",
    "billing",
}

TYPE_TO_TEMPLATE = {
    "cloudtrail": "cloudtrail",
    "iam_activity": "iam_activity",
    "guardduty": "guardduty",
    "cloudwatch": "cloudwatch",
    "access_key": "access_key",
    "billing": "billing",
}
