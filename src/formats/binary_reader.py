"""Big-endian reading shared by the BRLYT and BRLAN readers."""
from __future__ import annotations

import struct
from functools import lru_cache


class FormatError(ValueError):
    """A file is not in the format it is read as, or it is cut short."""


@lru_cache(maxsize=None)
def _struct(fmt: str) -> struct.Struct:
    return struct.Struct(">" + fmt)


class BinaryReader:
    """Reads big-endian values from a byte buffer, advancing its position."""

    __slots__ = ("data", "position")

    def __init__(self, data: bytes, position: int = 0):
        self.data = data
        self.position = position

    def seek(self, position: int) -> BinaryReader:
        self.position = position
        return self

    def skip(self, count: int) -> None:
        self.position += count

    def unpack(self, fmt: str) -> tuple:
        layout = _struct(fmt)
        try:
            values = layout.unpack_from(self.data, self.position)
        except struct.error:
            raise FormatError(f"The file ends inside a value at 0x{self.position:X}.") from None
        self.position += layout.size
        return values

    def u8(self) -> int:
        return self.unpack("B")[0]

    def u16(self) -> int:
        return self.unpack("H")[0]

    def u32(self) -> int:
        return self.unpack("I")[0]

    def vec2(self) -> tuple[float, float]:
        return self.unpack("2f")

    def vec3(self) -> tuple[float, float, float]:
        return self.unpack("3f")

    def color(self) -> tuple[int, int, int, int]:
        return self.unpack("4B")

    def bytes(self, count: int) -> bytes:
        end = self.position + count
        if count < 0 or end > len(self.data):
            raise FormatError(f"The file ends inside a {count} byte block at 0x{self.position:X}.")
        chunk = bytes(self.data[self.position:end])
        self.position = end
        return chunk

    def fourcc(self) -> str:
        return self.bytes(4).decode("latin-1")

    def cstring(self) -> str:
        """A NUL-terminated string of single-byte characters."""
        end = self.data.find(b"\0", self.position)
        if end < 0:
            raise FormatError(f"The string at 0x{self.position:X} is not terminated.")
        text = bytes(self.data[self.position:end]).decode("latin-1")
        self.position = end + 1
        return text

    def fixed_string(self, length: int) -> str:
        """A string padded with NULs to a fixed length."""
        return self.bytes(length).split(b"\0", 1)[0].decode("latin-1")
