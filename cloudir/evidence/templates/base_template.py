from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1200
HEIGHT = 760

COLORS = {
    "background": "#F6F7F8",
    "surface": "#FFFFFF",
    "surface_alt": "#FAFAFA",
    "card": "#FFFFFF",
    "border": "#D5DBDB",
    "border_dark": "#AAB7B8",
    "text": "#16191F",
    "muted": "#5F6B7A",
    "muted_2": "#7D8998",
    "aws_dark": "#232F3E",
    "aws_orange": "#FF9900",
    "blue": "#0972D3",
    "blue_soft": "#EAF4FF",
    "green": "#037F0C",
    "green_soft": "#E9F7EF",
    "red": "#D13212",
    "red_soft": "#FDEDE8",
    "yellow": "#B7791F",
    "yellow_soft": "#FFF4CC",
    "purple": "#6B46C1",
    "purple_soft": "#F3ECFF",
    "code_bg": "#F2F3F3",
    "table_header": "#F2F3F3",
    "table_row_alt": "#FAFAFA",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def load_mono_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue

    return load_font(size)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return bbox[2] - bbox[0]


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return bbox[3] - bbox[1]


def clean_text(value: Any, default: str = "Unknown") -> str:
    if value in [None, "", [], {}]:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text


def fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: Any,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    text = clean_text(text)

    if text_width(draw, text, font) <= max_width:
        return text

    ellipsis = "..."

    while text and text_width(draw, text + ellipsis, font) > max_width:
        text = text[:-1]

    return text + ellipsis if text else ellipsis


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: Any,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int = 3,
) -> list[str]:
    text = clean_text(text)

    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        current = word

        if text_width(draw, current, font) > max_width:
            current = fit_text_to_width(draw, current, font, max_width)

        if len(lines) >= max_lines:
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    lines = lines[:max_lines]

    if lines and len(" ".join(lines)) < len(text):
        lines[-1] = fit_text_to_width(draw, lines[-1], font, max_width - 12)

    return lines or ["Unknown"]


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: Any,
    max_width: int,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 6,
    max_lines: int = 3,
) -> int:
    x, y = xy
    lines = wrap_text_to_width(
        draw=draw,
        text=text,
        font=font,
        max_width=max_width,
        max_lines=max_lines,
    )

    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += font.size + line_gap

    return y


def create_canvas(title: str, subtitle: str, *, height: int = HEIGHT) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, height), COLORS["background"])
    draw = ImageDraw.Draw(image)

    # AWS-style dark top bar
    draw.rectangle((0, 0, WIDTH, 56), fill=COLORS["aws_dark"])
    draw.text((30, 16), "aws", fill=COLORS["aws_orange"], font=load_font(22, bold=True))
    draw.text((88, 18), "Console", fill="#FFFFFF", font=load_font(17, bold=True))
    draw.text((1015, 18), "ap-southeast-1", fill="#D5DBDB", font=load_font(14))
    draw.text((1120, 18), "Prod", fill="#D5DBDB", font=load_font(14))

    # Service header
    draw.text((46, 88), title, fill=COLORS["text"], font=load_font(30, bold=True))
    draw.text((48, 128), subtitle, fill=COLORS["muted"], font=load_font(16))
    draw.line((46, 153, WIDTH - 46, 153), fill=COLORS["border"], width=1)

    return image, draw


def draw_breadcrumb(draw: ImageDraw.ImageDraw, items: list[str], x: int = 46, y: int = 66) -> None:
    font = load_font(13)
    current_x = x

    for index, item in enumerate(items):
        color = COLORS["blue"] if index < len(items) - 1 else COLORS["muted"]
        draw.text((current_x, y), item, fill=color, font=font)
        current_x += text_width(draw, item, font) + 8

        if index < len(items) - 1:
            draw.text((current_x, y), "›", fill=COLORS["muted_2"], font=font)
            current_x += 16


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str | None = None,
) -> None:
    x1, y1, x2, y2 = box

    draw.rounded_rectangle(
        box,
        radius=14,
        fill=COLORS["card"],
        outline=COLORS["border"],
        width=1,
    )

    draw.text((x1 + 24, y1 + 18), title, fill=COLORS["text"], font=load_font(20, bold=True))

    if subtitle:
        draw.text((x1 + 24, y1 + 47), subtitle, fill=COLORS["muted"], font=load_font(13))
        line_y = y1 + 76
    else:
        line_y = y1 + 58

    draw.line((x1, line_y, x2, line_y), fill=COLORS["border"], width=1)


def draw_section_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x, y), text, fill=COLORS["text"], font=load_font(16, bold=True))


def draw_status_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: Any,
    tone: str = "blue",
) -> None:
    value = clean_text(text)

    palette = {
        "green": (COLORS["green_soft"], COLORS["green"]),
        "red": (COLORS["red_soft"], COLORS["red"]),
        "yellow": (COLORS["yellow_soft"], COLORS["yellow"]),
        "blue": (COLORS["blue_soft"], COLORS["blue"]),
        "purple": (COLORS["purple_soft"], COLORS["purple"]),
        "muted": (COLORS["code_bg"], COLORS["muted"]),
    }

    fill, outline = palette.get(tone, palette["blue"])
    font = load_font(13, bold=True)
    width = text_width(draw, value, font) + 28

    draw.rounded_rectangle(
        (x, y, x + width, y + 28),
        radius=14,
        fill=fill,
        outline=outline,
        width=1,
    )
    draw.text((x + 14, y + 7), value, fill=outline, font=font)


def draw_key_values(
    draw: ImageDraw.ImageDraw,
    start_x: int,
    start_y: int,
    rows: list[tuple[str, Any]],
    row_gap: int = 45,
    key_width: int = 210,
    value_width: int = 430,
) -> None:
    key_font = load_font(14, bold=True)
    value_font = load_font(14)
    y = start_y

    for key, value in rows:
        draw.text((start_x, y), str(key), fill=COLORS["muted"], font=key_font)

        draw_wrapped_text(
            draw=draw,
            xy=(start_x + key_width, y),
            text=clean_text(value),
            max_width=value_width,
            font=value_font,
            fill=COLORS["text"],
            line_gap=3,
            max_lines=2,
        )

        y += row_gap


def draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    headers: list[str],
    rows: list[list[Any]],
    widths: list[int],
    row_height: int = 46,
    header_height: int = 42,
) -> None:
    total_width = sum(widths)

    draw.rounded_rectangle(
        (x, y, x + total_width, y + header_height),
        radius=8,
        fill=COLORS["table_header"],
        outline=COLORS["border"],
        width=1,
    )

    header_font = load_font(13, bold=True)
    cell_font = load_font(13)

    current_x = x

    for header, width in zip(headers, widths):
        draw.text(
            (current_x + 12, y + 13),
            str(header),
            fill=COLORS["text"],
            font=header_font,
        )
        current_x += width

    y += header_height

    for index, row in enumerate(rows):
        current_x = x
        fill = COLORS["surface"] if index % 2 == 0 else COLORS["table_row_alt"]

        draw.rectangle(
            (x, y, x + total_width, y + row_height),
            fill=fill,
            outline=COLORS["border"],
            width=1,
        )

        for cell, width in zip(row, widths):
            text = fit_text_to_width(draw, clean_text(cell), cell_font, width - 24)
            draw.text(
                (current_x + 12, y + 15),
                text,
                fill=COLORS["text"],
                font=cell_font,
            )
            current_x += width

        y += row_height


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: Any,
    tone: str = "blue",
) -> None:
    x1, y1, x2, y2 = box

    tone_color = {
        "blue": COLORS["blue"],
        "green": COLORS["green"],
        "red": COLORS["red"],
        "yellow": COLORS["yellow"],
        "purple": COLORS["purple"],
    }.get(tone, COLORS["blue"])

    draw.rounded_rectangle(
        box,
        radius=12,
        fill=COLORS["surface_alt"],
        outline=COLORS["border"],
        width=1,
    )

    draw.text((x1 + 16, y1 + 14), label, fill=COLORS["muted"], font=load_font(13, bold=True))
    draw_wrapped_text(
        draw=draw,
        xy=(x1 + 16, y1 + 42),
        text=clean_text(value),
        max_width=(x2 - x1) - 32,
        font=load_font(18, bold=True),
        fill=tone_color,
        line_gap=4,
        max_lines=2,
    )


def draw_code_block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    title: str | None = None,
) -> None:
    x1, y1, x2, y2 = box

    draw.rounded_rectangle(
        box,
        radius=10,
        fill=COLORS["code_bg"],
        outline=COLORS["border"],
        width=1,
    )

    y = y1 + 14

    if title:
        draw.text((x1 + 16, y), title, fill=COLORS["muted"], font=load_font(12, bold=True))
        y += 28

    font = load_font(13)

    for line in lines:
        fitted = fit_text_to_width(draw, line, font, (x2 - x1) - 32)
        draw.text((x1 + 16, y), fitted, fill=COLORS["text"], font=font)
        y += 22

        if y > y2 - 24:
            break


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
