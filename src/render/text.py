"""Laying text out with a BRFNT font, with the metrics NW4R's TextWriter uses.

The font size scales glyphs by ``font_size.x / font.width`` across and ``font_size.y / font.height`` down.
Characters advance by their char width plus the character space; each line is the font's line feed plus
the line space high. A text box's text is drawn as one run: its glyphs in an image at the font's pixel
size, which the renderer scales and places in the pane. The glyphs keep the colours their sheet's texture
format gives them, as GX samples it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..formats import Brfnt, Glyph


@dataclass(frozen=True)
class TextRun:
    """Laid out text.

    ``width`` and ``height`` are the text's extent in layout units, which aligning it in the pane uses.
    ``texels`` holds the glyphs, a (rows, columns, 4) RGBA uint8 image, with ``scale_x`` and ``scale_y``
    layout units per pixel. Glyphs may reach past the extent, so the image starts ``left`` layout units
    right of the extent's left edge (usually a little left of it).
    """

    texels: np.ndarray
    scale_x: float
    scale_y: float
    width: float
    height: float
    left: float = 0.0

    @property
    def image_width(self) -> float:
        return self.texels.shape[1] * self.scale_x

    @property
    def image_height(self) -> float:
        return self.texels.shape[0] * self.scale_y


def line_alignment(text_alignment: int, text_position: int) -> int:
    """0 left, 1 center, 2 right. Alignment 0 follows the text's horizontal position in the pane."""
    return text_position % 3 if text_alignment == 0 else min(max(text_alignment - 1, 0), 2)


def layout_text(text: str, font: Brfnt, font_size: tuple[float, float], char_space: float, line_space: float,
                alignment: int) -> Optional[TextRun]:
    """Lay ``text`` out in lines; None when nothing would be drawn."""
    scale_x = font_size[0] / font.width if font.width and font_size[0] > 0 else 1.0
    scale_y = font_size[1] / font.height if font.height and font_size[1] > 0 else 1.0
    char_gap = char_space / scale_x  # in font pixels, like everything below
    line_pitch = font.line_feed + line_space / scale_y

    lines = [[font.glyph(font.glyph_index(char)) for char in line] for line in text.replace("\r", "").split("\n")]
    line_widths = [sum(glyph.widths.char_width for glyph in line) + char_gap * max(0, len(line) - 1)
                   for line in lines]
    width = max(line_widths, default=0.0)
    if width <= 0:
        return None
    height = len(lines) * line_pitch
    rows = max(height, (len(lines) - 1) * line_pitch + font.cell_height)

    placed: list[tuple[Glyph, float, float]] = []
    for line_index, (line, line_width) in enumerate(zip(lines, line_widths)):
        pen = (0.0, (width - line_width) / 2, width - line_width)[alignment]
        for glyph in line:
            placed.append((glyph, pen + glyph.widths.left, line_index * line_pitch))
            pen += glyph.widths.char_width + char_gap

    left = min(0.0, min((x for _, x, _ in placed), default=0.0))
    right = max(width, max((x + glyph.widths.glyph_width for glyph, x, _ in placed), default=0.0))
    texels = np.zeros((max(1, math.ceil(rows)), max(1, math.ceil(right - left)), 4), dtype=np.uint8)
    for glyph, x, y in placed:
        _blit_glyph(texels, font, glyph, round(x - left), round(y))

    return TextRun(texels, scale_x, scale_y, width * scale_x, height * scale_y, left * scale_x)


def _blit_glyph(texels: np.ndarray, font: Brfnt, glyph: Glyph, left: int, top: int) -> None:
    """Draw a glyph where it covers more than what is already there (glyphs may overlap)."""
    sheet = font.sheet(glyph.sheet)
    image = sheet[glyph.y:glyph.y + glyph.height, glyph.x:glyph.x + min(glyph.widths.glyph_width, glyph.width)]

    height, width = texels.shape[:2]
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + image.shape[1], width), min(top + image.shape[0], height)
    if x0 >= x1 or y0 >= y1:
        return
    source = image[y0 - top:y1 - top, x0 - left:x1 - left]
    target = texels[y0:y1, x0:x1]
    covers = source[..., 3] > target[..., 3]
    target[covers] = source[covers]
