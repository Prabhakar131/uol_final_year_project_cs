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


def _normalise_rows(rows: Any) -> list[list[str]]:
    if not isinstance(rows, list):
        return []

    output: list[list[str]] = []

    for row in rows[:5]:
        if isinstance(row, list):
            padded = [str(item) for item in row[:4]]

            while len(padded) < 4:
                padded.append("Unknown")

            output.append(padded)

        elif isinstance(row, dict):
            output.append(
                [
                    str(row.get("time", row.get("event_time", row.get("eventTime", row.get("timestamp", "Unknown"))))),
                    str(row.get("user", row.get("principal", row.get("actor", row.get("username", "Unknown"))))),
                    str(row.get("action", row.get("event_name", row.get("eventName", row.get("operation", "Unknown"))))),
                    str(row.get("source_ip", row.get("sourceIPAddress", row.get("ip", row.get("remote_ip", "Unknown"))))),
                ]
            )

    return output


def _time_span(rows: list[list[str]]) -> str:
    times = sorted(row[0] for row in rows if row[0][:4].isdigit())
    if not times:
        return "Time unknown"
    first, last = times[0].replace("T", " ")[:16], times[-1].replace("T", " ")[:16]
    return first if first == last else f"{first} to {last[11:] if last[:10] == first[:10] else last}"


def render_iam_activity(evidence: dict[str, Any], output_path: Path) -> None:
    facts = evidence["facts"]
    rows = _normalise_rows(facts.get("activity_rows", []))
    # Missing values render as Unknown; defaults must not add incident signals.
    principal = facts.get("principal") or (rows[0][1] if rows else "Unknown")
    row_source_ip = rows[0][3] if rows else "Unknown"
    source_ip = facts.get("source_ip") or row_source_ip

    if str(source_ip).startswith("192.0.2.") and row_source_ip:
        source_ip = row_source_ip

    mfa = facts.get("mfa", "Unknown")
    policy_change = facts.get("policy_change", "Unknown")
    access_key_status = facts.get("access_key_status", "Unknown")

    if str(access_key_status).lower() == "inactive" and any("createaccesskey" in str(row[2]).lower() for row in rows):
        access_key_status = "Key creation observed"

    image, draw = create_canvas(
        title="IAM",
        subtitle="Identity investigation · Production account",
    )
    draw.rectangle((0, 154, 1200, 760), fill="#F5F3FF")

    draw_breadcrumb(draw, ["IAM", "Users", str(principal), "Activity"])

    draw_panel(
        draw,
        (46, 172, 1154, 740),
        "User summary",
        "IAM identity profile with security credentials and recent activity",
    )

    # The real time span of the rows: a fixed "Last 60 minutes" label made older
    # baseline activity look like it happened during the incident.
    draw_status_pill(draw, 894, 184, _time_span(rows), tone="blue")
    draw_status_pill(draw, 894, 216, f"{len(rows)} events", tone="purple")

    label_font = load_font(12, bold=True)
    value_font = load_font(13)
    tab_font = load_font(13, bold=True)

    # IAM is rendered as an identity profile page with tabs and a security
    # credentials side panel, visually distinct from logs and audit events.
    draw.rounded_rectangle((78, 270, 362, 706), radius=8, fill=COLORS["surface_alt"], outline=COLORS["border"])
    draw.ellipse((102, 294, 158, 350), fill=COLORS["purple_soft"], outline=COLORS["purple"])
    draw.text((121, 310), "IAM", fill=COLORS["purple"], font=load_font(13, bold=True))
    draw.text((180, 296), "Principal", fill=COLORS["muted"], font=label_font)
    draw_wrapped_text(draw, (180, 320), principal, 150, load_font(16, bold=True), COLORS["text"], max_lines=2)

    draw.line((102, 375, 338, 375), fill=COLORS["border"])
    draw_key_values(
        draw=draw,
        start_x=102,
        start_y=398,
        rows=[
            ("MFA", mfa),
            ("Source IP", source_ip),
            ("Policy", policy_change),
            ("Access key", access_key_status),
        ],
        row_gap=46,
        key_width=86,
        value_width=125,
    )

    # No summary box here: a printed "risk flags" verdict let the judge copy a
    # conclusion, and a "last activity" box made the VLM skip the activity table.

    tab_y = 270
    tabs = [
        ("Permissions", False),
        ("Security credentials", True),
        ("Access Advisor", False),
        ("Activity", False),
    ]
    x = 392
    for tab, selected in tabs:
        width = 150 if tab != "Security credentials" else 195
        fill = COLORS["blue_soft"] if selected else COLORS["surface_alt"]
        outline = COLORS["blue"] if selected else COLORS["border"]
        draw.rounded_rectangle((x, tab_y, x + width, tab_y + 42), radius=8, fill=fill, outline=outline)
        draw.text((x + 14, tab_y + 13), tab, fill=COLORS["blue"] if selected else COLORS["muted"], font=tab_font)
        x += width + 8

    draw.rounded_rectangle((392, 332, 1118, 460), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((416, 354), "Security credentials", fill=COLORS["text"], font=load_font(18, bold=True))
    draw.text((416, 386), "Console password", fill=COLORS["muted"], font=label_font)
    draw.text((558, 386), "Enabled", fill=COLORS["green"], font=value_font)
    draw.text((416, 416), "Access key status", fill=COLORS["muted"], font=label_font)
    draw.text((558, 416), fit_text_to_width(draw, str(access_key_status), value_font, 250), fill=COLORS["text"], font=value_font)
    draw.text((818, 386), "MFA device", fill=COLORS["muted"], font=label_font)
    draw.text((930, 386), str(mfa), fill=COLORS["red"] if str(mfa).lower() in ["false", "no", "0"] else COLORS["green"], font=value_font)

    draw.text((392, 466), "Recent activity", fill=COLORS["text"], font=load_font(16, bold=True))
    draw.text((514, 469), f"{len(rows)} events, {_time_span(rows)}", fill=COLORS["muted"], font=value_font)

    draw_table(
        draw=draw,
        x=392,
        y=500,
        headers=["Time", "Principal", "Action", "Source IP"],
        rows=rows,
        widths=[145, 250, 205, 125],
        row_height=37,
        header_height=38,
    )

    save_image(image, output_path)
