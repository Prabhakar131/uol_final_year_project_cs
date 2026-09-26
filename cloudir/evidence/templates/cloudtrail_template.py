from __future__ import annotations

from pathlib import Path
from typing import Any

from cloudir.evidence.templates.base_template import (
    COLORS,
    create_canvas,
    draw_breadcrumb,
    draw_panel,
    draw_status_pill,
    draw_wrapped_text,
    fit_text_to_width,
    load_font,
    save_image,
)


def _normalise_related_events(rows: Any, facts: dict[str, Any]) -> list[list[str]]:
    if isinstance(rows, list) and rows:
        output: list[list[str]] = []

        for row in rows[:4]:
            if isinstance(row, list):
                padded = [str(item) for item in row[:5]]

                while len(padded) < 5:
                    padded.append("Unknown")

                output.append(padded)

            elif isinstance(row, dict):
                output.append(
                    [
                        str(row.get("time", row.get("event_time", row.get("eventTime", "Unknown")))),
                        str(row.get("event_name", row.get("eventName", row.get("action", "Unknown")))),
                        str(row.get("user", row.get("principal", row.get("actor", "Unknown")))),
                        str(row.get("source_ip", row.get("sourceIPAddress", row.get("ip", "Unknown")))),
                        str(row.get("result", row.get("status", row.get("mfa", "Unknown")))),
                    ]
                )

        return output

    # Only the selected event is known; do not draw invented surrounding activity.
    return [
        [
            str(facts.get("event_time", "Unknown")),
            str(facts.get("event_name", "Unknown")),
            str(facts.get("user", "Unknown")),
            str(facts.get("source_ip", "Unknown")),
            "Success",
        ]
    ]


def render_cloudtrail(evidence: dict[str, Any], output_path: Path) -> None:
    facts = evidence["facts"]

    event_name = facts.get("event_name", "Unknown")
    event_source = facts.get("event_source", "cloudtrail.amazonaws.com")
    user = facts.get("user", "Unknown")
    source_ip = facts.get("source_ip", "Unknown")
    mfa = facts.get("mfa", "Unknown")
    event_time = facts.get("event_time", "Unknown")
    region = facts.get("region", "Unknown")
    error_code = facts.get("error_code", "-")
    resources = facts.get("resources") or "None recorded"
    event_id = facts.get("event_id", "8f42b4ac-9f1b-4d1a-a3c8-5df2c9a71c03")
    user_agent = facts.get("user_agent", "signin.amazonaws.com")
    account_id = facts.get("recipient_account_id", "123456789012")
    request_parameters = facts.get("request_parameters", f"trailName=prod-org-trail")
    related_events = _normalise_related_events(facts.get("related_events"), facts)

    image, draw = create_canvas(
        title="CloudTrail",
        subtitle="Event history · Production account",
    )
    draw.rectangle((0, 154, 1200, 760), fill="#FFF7ED")

    draw_breadcrumb(draw, ["CloudTrail", "Event history", str(event_name)])

    draw_panel(
        draw,
        (46, 172, 1154, 704),
        "Event record",
        "CloudTrail Event history with selected API record details",
    )

    draw_status_pill(draw, 884, 184, "Management event", tone="blue")
    draw_status_pill(
        draw,
        884,
        211,
        "MFA: false" if str(mfa).lower() in ["false", "no", "0"] else f"MFA: {mfa}",
        tone="red" if str(mfa).lower() in ["false", "no", "0"] else "green",
    )

    heading_font = load_font(15, bold=True)
    small_font = load_font(12)
    row_font = load_font(13)
    mono_font = load_font(13)

    # CloudTrail is rendered as a split event browser: event list on the left,
    # selected API event details on the right.
    draw.rounded_rectangle((78, 270, 402, 674), radius=8, fill=COLORS["surface_alt"], outline=COLORS["border"])
    draw.text((98, 292), "Event history", fill=COLORS["text"], font=heading_font)
    draw.text((98, 316), "Latest events first", fill=COLORS["muted"], font=small_font)

    y = 348
    for index, row in enumerate(related_events):
        is_selected = str(row[1]).lower() == str(event_name).lower() or index == 0
        fill = COLORS["blue_soft"] if is_selected else COLORS["surface"]
        outline = COLORS["blue"] if is_selected else COLORS["border"]
        draw.rounded_rectangle((96, y, 384, y + 68), radius=8, fill=fill, outline=outline)
        draw.text((112, y + 10), fit_text_to_width(draw, row[1], heading_font, 150), fill=COLORS["blue"] if is_selected else COLORS["text"], font=heading_font)
        draw.text((112, y + 34), fit_text_to_width(draw, row[0], small_font, 120), fill=COLORS["muted"], font=small_font)
        draw.text((244, y + 34), fit_text_to_width(draw, row[4], small_font, 105), fill=COLORS["muted"], font=small_font)
        y += 78

    draw.rounded_rectangle((430, 270, 1118, 674), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.rectangle((430, 270, 1118, 322), fill="#F8FAFC", outline=COLORS["border"])
    draw.text((454, 288), str(event_name), fill=COLORS["text"], font=load_font(22, bold=True))
    draw.text((650, 292), fit_text_to_width(draw, str(event_time), row_font, 205), fill=COLORS["muted"], font=row_font)
    draw.text((884, 292), fit_text_to_width(draw, str(region), row_font, 100), fill=COLORS["blue"], font=row_font)

    detail_rows = [
        ("User identity", user),
        ("Source IP address", source_ip),
        ("Event source", event_source),
        ("User agent", user_agent),
        ("Recipient account", account_id),
        ("Error code", error_code),
        ("Event ID", event_id),
    ]

    y = 350
    for label, value in detail_rows:
        draw.text((454, y), label, fill=COLORS["muted"], font=load_font(12, bold=True))
        draw.text((620, y), fit_text_to_width(draw, str(value), row_font, 420), fill=COLORS["text"], font=row_font)
        y += 30

    draw.rounded_rectangle((454, 575, 746, 652), radius=8, fill=COLORS["code_bg"], outline=COLORS["border"])
    draw.text((470, 592), "Request parameters", fill=COLORS["muted"], font=load_font(12, bold=True))
    draw.text((470, 620), fit_text_to_width(draw, str(request_parameters), mono_font, 245), fill=COLORS["text"], font=mono_font)

    # Real CloudTrail shows the resources an event referenced, not an analyst's
    # verdict; a printed "investigation signal" let the judge copy a conclusion.
    draw.rounded_rectangle((770, 575, 1094, 652), radius=8, fill=COLORS["code_bg"], outline=COLORS["border"])
    draw.text((790, 592), "Resources referenced", fill=COLORS["muted"], font=load_font(12, bold=True))

    draw_wrapped_text(
        draw=draw,
        xy=(790, 620),
        text=str(resources),
        max_width=280,
        font=mono_font,
        fill=COLORS["text"],
        line_gap=4,
        max_lines=2,
    )

    save_image(image, output_path)
