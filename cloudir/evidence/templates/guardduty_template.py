from __future__ import annotations

from pathlib import Path
from typing import Any

from cloudir.evidence.templates.base_template import (
    COLORS,
    create_canvas,
    draw_breadcrumb,
    draw_key_values,
    draw_panel,
    draw_status_pill,
    draw_wrapped_text,
    load_font,
    save_image,
)


def _severity_tone(severity: Any) -> str:
    value = str(severity).lower()

    if "critical" in value or "high" in value or value in ["7", "8", "9", "10"]:
        return "red"

    if "medium" in value or value in ["4", "5", "6"]:
        return "yellow"

    if "low" in value or value in ["1", "2", "3"]:
        return "blue"

    return "muted"


def render_guardduty(evidence: dict[str, Any], output_path: Path) -> None:
    facts = evidence["facts"]

    finding_type = facts.get("finding_type", "Unknown")
    severity = facts.get("severity", "Unknown")
    resource = facts.get("resource", "Unknown")
    principal = facts.get("principal", "Unknown")
    remote_ip = facts.get("remote_ip", "Unknown")
    first_seen = facts.get("first_seen", "Unknown")
    last_seen = facts.get("last_seen", "Unknown")
    summary = facts.get("summary", "Suspicious activity detected.")

    severity_tone = _severity_tone(severity)

    image, draw = create_canvas(
        title="GuardDuty",
        subtitle="Finding details · Threat detection",
    )
    draw.rectangle((0, 154, 1200, 760), fill="#FEF2F2")

    draw_breadcrumb(draw, ["GuardDuty", "Findings", str(finding_type)])

    draw_panel(
        draw,
        (46, 172, 1154, 704),
        "Finding overview",
        "GuardDuty finding associated with suspicious AWS account activity",
    )

    draw_status_pill(draw, 930, 184, "Active", tone="red")
    draw_status_pill(draw, 930, 216, f"Severity: {severity}", tone=severity_tone)

    label_font = load_font(12, bold=True)
    value_font = load_font(13)

    draw.rounded_rectangle((78, 270, 1118, 392), radius=8, fill="#7F1D1D", outline="#991B1B")
    draw.text((104, 294), "ACTIVE FINDING", fill="#FECACA", font=label_font)
    draw_wrapped_text(
        draw=draw,
        xy=(104, 322),
        text=finding_type,
        max_width=650,
        font=load_font(23, bold=True),
        fill="#FFFFFF",
        line_gap=5,
        max_lines=2,
    )
    draw.rounded_rectangle((884, 302, 1070, 346), radius=22, fill="#FEE2E2", outline="#FCA5A5")
    draw.text((910, 315), f"Severity {severity}", fill=COLORS[severity_tone] if severity_tone in COLORS else COLORS["red"], font=load_font(15, bold=True))

    draw_panel(
        draw,
        (78, 420, 690, 680),
        "Affected resource",
        "Entity and network context",
    )

    draw_key_values(
        draw=draw,
        start_x=106,
        start_y=500,
        rows=[
            ("Principal", principal),
            ("Resource", resource),
            ("Remote IP", remote_ip),
            ("First seen", first_seen),
            ("Last seen", last_seen),
        ],
        row_gap=34,
        key_width=112,
        value_width=390,
    )

    draw.rounded_rectangle((718, 420, 1118, 530), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((742, 442), "Evidence", fill=COLORS["text"], font=load_font(18, bold=True))
    draw.text((742, 480), "Service", fill=COLORS["muted"], font=label_font)
    draw.text((832, 480), "GuardDuty", fill=COLORS["text"], font=value_font)
    draw.text((742, 506), "Status", fill=COLORS["muted"], font=label_font)
    draw.text((832, 506), "Active", fill=COLORS["red"], font=value_font)

    draw.rounded_rectangle((718, 552, 1118, 680), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((742, 574), "Finding summary", fill=COLORS["text"], font=load_font(18, bold=True))
    draw_wrapped_text(
        draw=draw,
        xy=(742, 616),
        text=summary,
        max_width=330,
        font=value_font,
        fill=COLORS["text"],
        line_gap=5,
        max_lines=3,
    )

    save_image(image, output_path)
