"""TPL textures and GX texel data, decoded with TPLLib into RGBA pixels as GX samples them."""
from __future__ import annotations

import struct
from typing import Optional, Sequence

import numpy as np

from . import TPLLib
from .binary_reader import FormatError

TPL_MAGIC = 0x0020AF30
INDEXED_FORMATS = (TPLLib.CI4, TPLLib.CI8, TPLLib.CI14X2)
INTENSITY_FORMATS = (TPLLib.I4, TPLLib.I8)
PALETTE_IA8, PALETTE_RGB565, PALETTE_RGB5A3 = 0, 1, 2


def read_tpl(data: bytes, index: int = 0) -> np.ndarray:
    """One image of a TPL file as a read-only (height, width, 4) RGBA uint8 array."""
    try:
        magic, count, table = struct.unpack_from(">III", data)
        if magic != TPL_MAGIC or not 0 <= index < count:
            raise FormatError(f"This is not a TPL file with an image {index}.")
        image, palette = struct.unpack_from(">II", data, table + 8 * index)
        height, width, fmt, offset = struct.unpack_from(">HHII", data, image)
        colors = None
        if palette:
            entries, _, _, palette_format, palette_offset = struct.unpack_from(">HBBII", data, palette)
            colors = _decode_palette(palette_format, data[palette_offset:palette_offset + 2 * entries])
    except struct.error:
        raise FormatError("This TPL file is cut short.") from None
    return decode_texels(fmt, data[offset:], width, height, colors)


def decode_texels(fmt: int, data: bytes, width: int, height: int,
                  palette: Optional[Sequence[tuple[int, int, int, int]]] = None) -> np.ndarray:
    """Tiled GX texel data as a read-only (height, width, 4) RGBA uint8 array.

    GX samples I4 and I8 textures with the intensity in every channel, alpha included, while TPLLib gives
    them an opaque alpha; the alpha is set to the intensity here.
    """
    try:
        decoder = TPLLib.getDecoder(fmt)
    except ValueError:
        raise FormatError(f"Unsupported GX texture format {fmt}.") from None
    if fmt in INDEXED_FORMATS and palette is None:
        raise FormatError("The indexed texture has no palette.")
    try:
        bgra = decoder(bytes(data), width, height, list(palette) if palette is not None else None).run()
    except (TypeError, ValueError) as error:
        raise FormatError(str(error)) from None

    pixels = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)[..., [2, 1, 0, 3]]
    if fmt in INTENSITY_FORMATS:
        pixels[..., 3] = pixels[..., 0]
    pixels.flags.writeable = False
    return pixels


def _decode_palette(palette_format: int, data: bytes) -> list[tuple[int, int, int, int]]:
    """TLUT entries as the BGRA tuples TPLLib's indexed decoders look colours up in."""
    entries = []
    for (value,) in struct.iter_unpack(">H", data[:len(data) & ~1]):
        if palette_format == PALETTE_IA8:  # AAAAAAAA IIIIIIII, like IA8 texels
            intensity = value & 0xFF
            entries.append((intensity, intensity, intensity, value >> 8))
        elif palette_format == PALETTE_RGB565:
            entries.append(TPLLib._rgb565(value))
        elif palette_format == PALETTE_RGB5A3:
            entries.append(TPLLib._rgb5a3(value))
        else:
            raise FormatError(f"Unsupported TPL palette format {palette_format}.")
    return entries
