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
    draw_table,
    draw_wrapped_text,
    fit_text_to_width,
    load_font,
    save_image,
)


def _status_tone(status: Any) -> str:
    value = str(status).lower()

    if "active" in value:
        return "red"

    if "inactive" in value or "disabled" in value:
        return "green"

    return "muted"


def render_access_key(evidence: dict[str, Any], output_path: Path) -> None:
    facts = evidence["facts"]

    access_key_id = facts.get("access_key_id", "Unknown")
    owner = facts.get("owner", "Unknown")
    status = facts.get("status", "Unknown")
    last_used_service = facts.get("last_used_service", "Unknown")
    last_used_region = facts.get("last_used_region", "Unknown")
    last_used_time = facts.get("last_used_time", "Unknown")
    source_ip = facts.get("source_ip", "Unknown")

    image, draw = create_canvas(
        title="IAM",
        subtitle="Access key activity · Credential review",
    )
    draw.rectangle((0, 154, 1200, 760), fill="#ECFDF5")

    draw_breadcrumb(draw, ["IAM", "Users", str(owner), "Security credentials"])

    draw_panel(
        draw,
        (46, 172, 1154, 704),
        "Access key details",
        "Credential usage information for the selected IAM user",
    )

    draw_status_pill(draw, 900, 184, f"Status: {status}", tone=_status_tone(status))
    draw_status_pill(draw, 900, 216, "Credential review", tone="purple")

    label_font = load_font(12, bold=True)
    value_font = load_font(13)

    draw.rounded_rectangle((78, 270, 1118, 350), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((102, 292), "Security credentials", fill=COLORS["text"], font=load_font(20, bold=True))
    draw.text((102, 322), fit_text_to_width(draw, str(owner), value_font, 450), fill=COLORS["muted"], font=value_font)

    draw_table(
        draw=draw,
        x=78,
        y=385,
        headers=["Access key ID", "Status", "Last used service", "Region", "Last used"],
        rows=[
            [access_key_id, status, last_used_service, last_used_region, last_used_time],
        ],
        widths=[270, 120, 240, 160, 250],
        row_height=54,
        header_height=42,
    )

    draw_key_values(
        draw=draw,
        start_x=102,
        start_y=520,
        rows=[
            ("Owner", owner),
            ("Source IP", source_ip),
            ("Rotation status", "Review required"),
            ("Console user", owner),
        ],
        row_gap=34,
        key_width=130,
        value_width=390,
    )

    draw.rounded_rectangle((700, 500, 1118, 675), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((724, 524), "Credential risk", fill=COLORS["text"], font=load_font(19, bold=True))
    draw.text((724, 556), "Programmatic access review", fill=COLORS["muted"], font=value_font)

    draw_wrapped_text(
        draw=draw,
        xy=(724, 596),
        text="An active access key used near the incident window can indicate programmatic access or credential abuse. Compare the source IP, service, region, and owner with CloudTrail events.",
        max_width=340,
        font=value_font,
        fill=COLORS["text"],
        line_gap=5,
        max_lines=4,
    )

    save_image(image, output_path)
