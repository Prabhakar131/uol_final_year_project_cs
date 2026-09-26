from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL.PngImagePlugin import PngInfo

from cloudir.evidence.templates.base_template import (
    COLORS, create_canvas, draw_breadcrumb, load_font, load_mono_font, text_width,
)
from cloudir.evidence_core.cloudwatch_facts import normalise_cloudwatch_facts


def _wrap_complete(draw, value: Any, font, width: int) -> list[str]:
    """Wrap without discarding long identifiers or adding ellipses."""
    text = "Unknown" if value is None or value == "" else str(value)
    lines = []
    for paragraph in text.split("\n"):
        remaining = paragraph
        if not remaining:
            lines.append("")
        while remaining:
            if text_width(draw, remaining, font) <= width:
                lines.append(remaining)
                break
            low, high = 1, len(remaining)
            while low < high:
                mid = (low + high + 1) // 2
                if text_width(draw, remaining[:mid], font) <= width:
                    low = mid
                else:
                    high = mid - 1
            cut = remaining.rfind(" ", 0, low + 1) + 1
            cut = cut if cut > 0 else low
            lines.append(remaining[:cut])
            remaining = remaining[cut:]
    return lines or ["Unknown"]


def render_cloudwatch(evidence: dict[str, Any], output_path: Path) -> None:
    facts = normalise_cloudwatch_facts(evidence["facts"])
    rows = facts["log_rows"]
    # Measure every line before allocating the final canvas. No row cap or
    # fixed-height table can silently omit content.
    _, measure = create_canvas("CloudWatch", "Logs Insights")
    body_font = load_font(16)
    timestamp_font = load_font(18)
    # Logs Insights shows results in monospace. It also keeps "rn" visibly
    # distinct from "m" in ARNs for the VLM, which misread proportional text.
    cell_font = load_mono_font(16)
    label_font = load_font(15, bold=True)
    line_height = 24
    query_lines = _wrap_complete(measure, facts["query"], body_font, 1010)
    count_top = 226
    query_top = 260
    query_bottom = query_top + 50 + line_height * len(query_lines)
    metadata_top = query_bottom + 20
    metadata_lines = []
    for label, key in (("Log group", "log_group"), ("Time range", "time_range"),
                       ("Status", "alarm_state"), ("Scanned", "scanned_bytes")):
        metadata_lines.extend(_wrap_complete(measure, f"{label}: {facts[key]}", body_font, 1010))
    metadata_bottom = metadata_top + 24 + line_height * len(metadata_lines)
    table_top = metadata_bottom + 20
    fonts = (timestamp_font, cell_font, cell_font)
    cells = [tuple(_wrap_complete(measure, value, font, width)
                   for value, font, width in zip(row, fonts, (210, 225, 565))) for row in rows]
    row_heights = [max(len(cell) for cell in row) * line_height + 20 for row in cells]
    table_bottom = table_top + 40 + (sum(row_heights) if rows else 56)
    height = max(780, table_bottom + 54)
    image, draw = create_canvas("CloudWatch", "Logs Insights · Query results", height=height)
    draw.rectangle((0, 154, 1200, height), fill="#EFF6FF")
    draw_breadcrumb(draw, ["CloudWatch", "Logs Insights"])
    draw.rounded_rectangle((46, 172, 1154, height - 20), radius=12,
                           fill=COLORS["surface"], outline=COLORS["border"])
    draw.text((70, 190), "Logs Insights query", font=load_font(22, bold=True), fill=COLORS["text"])
    # The total badge and displayed count are explicitly different concepts.
    count_text = f"Matched records: {facts['matched_records']}    |    Displayed rows: {len(rows)}"
    draw.text((70, count_top), count_text, font=body_font, fill=COLORS["muted"])
    draw.rounded_rectangle((70, query_top, 1130, query_bottom), radius=8, fill="#111827")
    draw.text((86, query_top + 12), "Query editor", font=label_font, fill="#D1D5DB")
    for i, line in enumerate(query_lines):
        draw.text((86, query_top + 38 + i * line_height), line, font=body_font, fill="#E5E7EB")
    draw.rounded_rectangle((70, metadata_top, 1130, metadata_bottom), radius=6,
                           fill=COLORS["surface_alt"], outline=COLORS["border"])
    for i, line in enumerate(metadata_lines):
        draw.text((86, metadata_top + 12 + i * line_height), line, font=body_font, fill=COLORS["text"])
    draw.rectangle((70, table_top, 1130, table_bottom), fill="#0B1220")
    draw.rectangle((70, table_top, 1130, table_top + 40), fill="#1F2937")
    columns = (84, 309, 554)
    for title, x in zip(("@timestamp", "@logStream", "@message"), columns):
        draw.text((x, table_top + 12), title, font=label_font, fill="#D1D5DB")
    y = table_top + 40
    for index, (row, row_height) in enumerate(zip(cells, row_heights)):
        if index % 2:
            draw.rectangle((71, y, 1129, y + row_height), fill="#162033")
        for x, lines, color, font in zip(columns, row, ("#BFDBFE", "#C4B5FD", "#E5E7EB"), fonts):
            for i, line in enumerate(lines):
                draw.text((x, y + 10 + i * line_height), line, font=font, fill=color)
        y += row_height
        draw.line((70, y, 1130, y), fill="#374151")
    if not rows:
        draw.text((84, y + 16), "No rows supplied for display.", font=body_font, fill="#E5E7EB")
    # Layout metadata contains coordinates only, never fact text or answer labels.
    # The box spans the matched-records line, query editor, metadata and table,
    # so the VLM can enlarge exactly the card content it transcribes.
    metadata = PngInfo()
    metadata.add_text("cloudir_cloudwatch_detail_box", json.dumps([70, count_top - 10, 1130, table_bottom]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, pnginfo=metadata)
