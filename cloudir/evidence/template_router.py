from __future__ import annotations

from pathlib import Path
from typing import Any

from cloudir.evidence.templates.access_key_template import render_access_key
from cloudir.evidence.templates.billing_template import render_billing
from cloudir.evidence.templates.cloudtrail_template import render_cloudtrail
from cloudir.evidence.templates.cloudwatch_template import render_cloudwatch
from cloudir.evidence.templates.guardduty_template import render_guardduty
from cloudir.evidence.templates.iam_activity_template import render_iam_activity


VALID_RENDERERS = {
    "cloudtrail": render_cloudtrail,
    "iam_activity": render_iam_activity,
    "guardduty": render_guardduty,
    "cloudwatch": render_cloudwatch,
    "access_key": render_access_key,
    "billing": render_billing,
}


def render_evidence_image(evidence: dict[str, Any], output_path: Path) -> None:
    template = evidence.get("template")
    evidence_id = evidence.get("id", "unknown")
    evidence_type = evidence.get("type", "unknown")

    renderer = VALID_RENDERERS.get(template)

    if renderer is None:
        raise ValueError(
            f"Unknown evidence template: {template}. "
            f"Evidence id={evidence_id}, type={evidence_type}. "
            f"Allowed templates: {sorted(VALID_RENDERERS.keys())}"
        )

    print(
        f"[Template Router] evidence_id={evidence_id} "
        f"type={evidence_type} template={template} "
        f"renderer={renderer.__name__}"
    )

    renderer(evidence, output_path)