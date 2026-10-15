"""Wii BRFNT fonts: metrics, character map and glyph sheets."""
from __future__ import annotations

import struct
from os import PathLike
from typing import NamedTuple, Optional, Union

import numpy as np

from .binary_reader import FormatError
from .tpl import decode_texels

SHEET_FORMAT_MASK = 0x7FFF  # the top bit flags compressed sheets


class CharWidths(NamedTuple):
    """A glyph's horizontal metrics in font pixels."""

    left: int  # blank space before the glyph image
    glyph_width: int
    char_width: int  # advance to the next character


class Glyph(NamedTuple):
    """Where a glyph's cell is on its sheet, and its metrics."""

    sheet: int
    x: int
    y: int
    width: int
    height: int
    widths: CharWidths


class Brfnt:
    def __init__(self, data: bytes, source: str = "<memory>"):
        self.source = source
        try:
            self._read(bytes(data))
        except struct.error as error:
            raise FormatError(f"{source} is damaged: {error}") from None

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> Brfnt:
        with open(path, "rb") as file:
            return cls(file.read(), str(path))

    def __repr__(self) -> str:
        return f"<Brfnt {self.source!r}>"

    def glyph_index(self, char: Union[str, int]) -> int:
        """The glyph a character draws with, or the replacement glyph when the font lacks it."""
        code = ord(char) if isinstance(char, str) else char
        return self._codes.get(code, self.alternate_glyph)

    def widths(self, glyph_index: int) -> CharWidths:
        return self._widths.get(glyph_index, self.default_widths)

    def char_widths(self, char: Union[str, int]) -> CharWidths:
        return self.widths(self.glyph_index(char))

    def glyph(self, glyph_index: int) -> Glyph:
        sheet, cell = divmod(glyph_index, self.sheet_columns * self.sheet_rows)
        row, column = divmod(cell, self.sheet_columns)
        # NW4R keeps a one pixel line around every cell.
        return Glyph(sheet, 1 + column * (self.cell_width + 1), 1 + row * (self.cell_height + 1),
                     self.cell_width, self.cell_height, self.widths(glyph_index))

    def sheet(self, index: int) -> np.ndarray:
        """A glyph sheet as a read-only (height, width, 4) RGBA array, decoded on first use."""
        if self._sheets[index] is None:
            self._sheets[index] = decode_texels(self.sheet_format, self._sheet_data[index],
                                                self.sheet_width, self.sheet_height)
        return self._sheets[index]

    def _read(self, data: bytes) -> None:
        if data[:4] != b"RFNT" or len(data) < 0x10:
            raise FormatError(f"{self.source} is not a BRFNT font.")
        finf = struct.unpack_from(">H", data, 0xC)[0]
        if data[finf:finf + 4] != b"FINF":
            raise FormatError(f"{self.source} has no FINF block.")
        (self.line_feed, self.alternate_glyph, left, glyph_width, char_width, tglp, cwdh, cmap, self.height,
         self.width) = struct.unpack_from(">xbHbBbxIIIBB", data, finf + 8)
        self.default_widths = CharWidths(left, glyph_width, char_width)
        if data[tglp - 8:tglp - 4] != b"TGLP":
            raise FormatError(f"{self.source} has no TGLP block.")
        (self.cell_width, self.cell_height, sheet_size, sheet_count, sheet_format, self.sheet_columns,
         self.sheet_rows, self.sheet_width, self.sheet_height, sheet_offset) = struct.unpack_from(
            ">BBxxI6HI", data, tglp)
        if sheet_offset + sheet_size * sheet_count > len(data):
            raise FormatError(f"{self.source} ends inside its glyph sheets.")
        if self.sheet_columns <= 0 or self.sheet_rows <= 0:
            raise FormatError(f"{self.source} has an empty glyph sheet layout.")
        self.sheet_format = sheet_format & SHEET_FORMAT_MASK
        self._sheet_data = [data[sheet_offset + index * sheet_size:sheet_offset + (index + 1) * sheet_size]
                            for index in range(sheet_count)]
        self._sheets: list[Optional[np.ndarray]] = [None] * sheet_count
        self._widths = _read_widths(data, cwdh)
        self._codes = _read_codes(data, cmap)


def _read_widths(data: bytes, link: int) -> dict[int, CharWidths]:
    """Glyph metrics by glyph index from the CWDH chain."""
    widths = {}
    while link:
        first, last, link_next = struct.unpack_from(">HHI", data, link)
        for index in range(first, last + 1):
            widths[index] = CharWidths(*struct.unpack_from(">bBb", data, link + 8 + 3 * (index - first)))
        link = link_next
    return widths


def _read_codes(data: bytes, link: int) -> dict[int, int]:
    """Glyph indices by character code from the CMAP chain."""
    codes = {}
    while link:
        first, last, method, _, link_next = struct.unpack_from(">4HI", data, link)
        table = link + 12
        if method == 0:
            base, = struct.unpack_from(">H", data, table)
            codes.update((code, base + code - first) for code in range(first, last + 1))
        elif method == 1:
            indices = struct.unpack_from(f">{last - first + 1}H", data, table)
            codes.update((code, index) for code, index in zip(range(first, last + 1), indices) if index != 0xFFFF)
        elif method == 2:
            count, = struct.unpack_from(">H", data, table)
            pairs = struct.unpack_from(f">{2 * count}H", data, table + 2)
            codes.update(zip(pairs[0::2], pairs[1::2]))
        else:
            raise FormatError(f"Unknown CMAP method {method}.")
        link = link_next
    return codes
