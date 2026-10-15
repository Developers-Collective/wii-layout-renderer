"""NW4R layout animations (BRLAN), read into an immutable model."""
from __future__ import annotations

from dataclasses import dataclass

from .binary_reader import BinaryReader, FormatError

STEP_CURVE = 1  # otherwise Hermite

PANE_CONTENT = 0
MATERIAL_CONTENT = 1


@dataclass(frozen=True)
class AnimationKey:
    frame: float
    value: float
    slope: float = 0.0


@dataclass(frozen=True)
class AnimationTarget:
    """The keys of one animated value, e.g. the X translation (RLPA target 0)."""

    index: int  # which texture map, SRT or frame of the content the value belongs to
    target: int  # which value, its meaning depends on the AnimationInfo kind
    curve_type: int
    keys: tuple[AnimationKey, ...]

    @property
    def is_step(self) -> bool:
        return self.curve_type == STEP_CURVE


@dataclass(frozen=True)
class AnimationInfo:
    """One kind of animation on a pane or material: RLPA, RLVI, RLVC, RLTS, RLTP, RLMC, ..."""

    kind: str
    targets: tuple[AnimationTarget, ...]


@dataclass(frozen=True)
class AnimationContent:
    """The animations of the pane or material with this name."""

    name: str
    type: int  # PANE_CONTENT or MATERIAL_CONTENT
    infos: tuple[AnimationInfo, ...]


@dataclass(frozen=True)
class LayoutAnimation:
    frame_size: int
    files: tuple[str, ...]  # texture names that RLTP keys index
    contents: tuple[AnimationContent, ...]


def read_brlan(data: bytes) -> LayoutAnimation:
    reader = BinaryReader(data)
    if reader.fourcc() != "RLAN":
        raise FormatError("Not a BRLAN animation.")
    reader.skip(4)  # byte order mark, version
    file_size = reader.u32()
    header_size = reader.u16()

    animation = LayoutAnimation(0, (), ())
    position = header_size
    end = min(file_size, len(data))
    while position + 8 <= end:
        reader.seek(position)
        kind = reader.fourcc()
        size = reader.u32()
        if size < 8:
            break
        if kind == "pai1":
            animation = _read_info(reader, position)
        position += size
    return animation


def _read_info(reader: BinaryReader, block: int) -> LayoutAnimation:
    reader.seek(block + 8)
    frame_size, file_count, content_count, content_table = reader.unpack("HxxHHI")

    file_table = block + 20
    file_offsets = reader.seek(file_table).unpack(f"{file_count}I")
    files = tuple(reader.seek(file_table + offset).cstring() for offset in file_offsets)

    content_offsets = reader.seek(block + content_table).unpack(f"{content_count}I")
    contents = tuple(_read_content(reader, block + offset) for offset in content_offsets)
    return LayoutAnimation(frame_size, files, contents)


def _read_content(reader: BinaryReader, start: int) -> AnimationContent:
    reader.seek(start)
    name = reader.fixed_string(20)
    count, content_type = reader.unpack("BBxx")
    offsets = reader.unpack(f"{count}I")
    return AnimationContent(name, content_type, tuple(_read_animation_info(reader, start + offset) for offset in offsets))


def _read_animation_info(reader: BinaryReader, start: int) -> AnimationInfo:
    reader.seek(start)
    kind = reader.fourcc()
    count = reader.u8()
    reader.skip(3)
    offsets = reader.unpack(f"{count}I")
    return AnimationInfo(kind, tuple(_read_target(reader, start + offset) for offset in offsets))


def _read_target(reader: BinaryReader, start: int) -> AnimationTarget:
    reader.seek(start)
    index, target, curve_type, key_count, keys_offset = reader.unpack("BBBxHxxI")
    reader.seek(start + keys_offset)
    if curve_type == STEP_CURVE:
        keys = tuple(AnimationKey(frame, float(value)) for frame, value in
                     (reader.unpack("fHxx") for _ in range(key_count)))
    else:
        keys = tuple(AnimationKey(*reader.unpack("3f")) for _ in range(key_count))
    return AnimationTarget(index, target, curve_type, keys)
