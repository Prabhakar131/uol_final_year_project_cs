from __future__ import annotations

from pathlib import Path
from typing import Any

from cloudir.evidence.templates.base_template import (
    COLORS,
    create_canvas,
    draw_panel,
    draw_status_pill,
    draw_wrapped_text,
    fit_text_to_width,
    load_font,
    save_image,
)


def render_billing(evidence: dict[str, Any], output_path: Path) -> None:
    facts = evidence["facts"]
    current_spend = facts.get("current_spend", "Unknown")
    previous_average = facts.get("previous_average", "Unknown")
    largest_service = facts.get("largest_service", "Unknown")
    region = facts.get("region", "Unknown")
    change = facts.get("change", "Unknown")

    image, draw = create_canvas(
        title="AWS Billing",
        subtitle="Cost Explorer · Last 24 Hours",
    )
    draw.rectangle((0, 154, 1200, 760), fill="#FFFBEB")

    draw_panel(draw, (46, 172, 1154, 704), "Cost Explorer", "Spend anomaly view for the investigation window")

    draw_status_pill(draw, 900, 184, "Last 24 hours", tone="blue")
    draw_status_pill(draw, 900, 216, f"Change: {change}", tone="red")

    label_font = load_font(12, bold=True)
    value_font = load_font(14)

    draw.rounded_rectangle((78, 270, 1118, 365), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((104, 292), "Estimated current spend", fill=COLORS["muted"], font=label_font)
    draw.text((104, 320), str(current_spend), fill=COLORS["yellow"], font=load_font(28, bold=True))
    draw.text((410, 292), "Previous average", fill=COLORS["muted"], font=label_font)
    draw.text((410, 322), str(previous_average), fill=COLORS["text"], font=load_font(18, bold=True))
    draw.text((684, 292), "Change", fill=COLORS["muted"], font=label_font)
    draw.text((684, 322), str(change), fill=COLORS["red"], font=load_font(20, bold=True))
    draw.text((910, 292), "Region", fill=COLORS["muted"], font=label_font)
    draw.text((910, 322), str(region), fill=COLORS["blue"], font=load_font(18, bold=True))

    chart_box = (78, 405, 720, 675)
    draw.rounded_rectangle(chart_box, radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((104, 428), "Daily spend trend", fill=COLORS["text"], font=load_font(18, bold=True))
    draw.line((126, 620, 680, 620), fill=COLORS["border_dark"], width=1)
    draw.line((126, 470, 126, 620), fill=COLORS["border_dark"], width=1)

    bars = [44, 48, 42, 50, 46, 92, 150]
    x = 168
    for index, bar_height in enumerate(bars):
        color = COLORS["yellow"] if index < 5 else COLORS["red"]
        draw.rounded_rectangle((x, 620 - bar_height, x + 42, 620), radius=5, fill=color)
        x += 70
    draw.text((146, 638), "Previous baseline", fill=COLORS["muted"], font=load_font(12))
    draw.text((562, 638), "Incident window", fill=COLORS["red"], font=load_font(12, bold=True))

    draw.rounded_rectangle((750, 405, 1118, 675), radius=8, fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((774, 428), "Service cost driver", fill=COLORS["text"], font=load_font(18, bold=True))
    draw.text((774, 466), "Largest service", fill=COLORS["muted"], font=label_font)
    draw.text((774, 494), fit_text_to_width(draw, str(largest_service), load_font(22, bold=True), 290), fill=COLORS["yellow"], font=load_font(22, bold=True))
    draw.text((774, 540), "Investigation note", fill=COLORS["muted"], font=label_font)
    note = "A sudden spend increase can indicate resource abuse after credential compromise. Correlate service, region, and timing with IAM and CloudTrail evidence."
    draw_wrapped_text(draw, (774, 568), note, 290, value_font, COLORS["text"], line_gap=5, max_lines=4)

    save_image(image, output_path)
