"""NW4R layouts (BRLYT), read into an immutable model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .binary_reader import BinaryReader, FormatError

Color = tuple[int, int, int, int]
Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class TexMap:
    texture_index: int
    wrap_s: int  # 0 clamp, 1 repeat, 2 mirror
    wrap_t: int
    min_linear: bool  # linear (True) or nearest (False) filtering when the texture is minified
    mag_linear: bool  # ... and when it is magnified

    @classmethod
    def read(cls, reader: BinaryReader) -> TexMap:
        texture_index, wrap_s_filter, wrap_t_filter = reader.unpack("HBB")
        # NW4R stores the GX filter minus GX_LINEAR in the bits above the wrap mode.
        return cls(texture_index, wrap_s_filter & 3, wrap_t_filter & 3,
                   min_linear=(wrap_s_filter >> 2) & 7 != 7, mag_linear=(wrap_t_filter >> 2) & 1 != 1)


@dataclass(frozen=True)
class TexSrt:
    translate: Vec2
    rotate: float
    scale: Vec2

    @classmethod
    def read(cls, reader: BinaryReader) -> TexSrt:
        tx, ty, rotate, sx, sy = reader.unpack("5f")
        return cls((tx, ty), rotate, (sx, sy))


@dataclass(frozen=True)
class AlphaCompare:
    comp0: int  # GXCompare
    comp1: int
    op: int  # GXAlphaOp: 0 and, 1 or, 2 xor, 3 xnor
    ref0: int
    ref1: int


@dataclass(frozen=True)
class BlendMode:
    type: int  # GXBlendMode: 0 none, 1 blend, 2 logic, 3 subtract
    src_factor: int
    dst_factor: int
    logic_op: int


@dataclass(frozen=True)
class Material:
    name: str
    tev_colors: tuple[Color, Color, Color]  # signed 10-bit channels; [0] is black, [1] white
    tex_maps: tuple[TexMap, ...]
    tex_srts: tuple[TexSrt, ...]
    alpha_compare: Optional[AlphaCompare]
    blend_mode: Optional[BlendMode]

    @classmethod
    def read(cls, reader: BinaryReader) -> Material:
        name = reader.fixed_string(20)
        tev_colors = tuple(reader.unpack("4h") for _ in range(3))
        reader.skip(16)  # TEV constant colours
        flags = reader.u32()

        def bits(position: int, length: int) -> int:
            return (flags >> position) & ((1 << length) - 1)

        tex_maps = tuple(TexMap.read(reader) for _ in range(bits(0, 4)))
        tex_srts = tuple(TexSrt.read(reader) for _ in range(bits(4, 4)))
        reader.skip(bits(8, 4) * 4)  # texture coordinate generators
        reader.skip(bits(25, 1) * 4 + bits(27, 1) * 4)  # channel control, material colour
        if bits(12, 1):
            reader.skip(4)  # TEV swap table
        reader.skip(bits(13, 2) * 20)  # indirect texture SRTs
        reader.skip(bits(15, 3) * 4)  # indirect stages
        reader.skip(bits(18, 5) * 16)  # TEV stages
        alpha_compare = None
        if bits(23, 1):
            comp, op, ref0, ref1 = reader.unpack("4B")
            alpha_compare = AlphaCompare(comp & 0xF, comp >> 4, op, ref0, ref1)
        blend_mode = BlendMode(*reader.unpack("4B")) if bits(24, 1) else None
        return cls(name, tev_colors, tex_maps, tex_srts, alpha_compare, blend_mode)


@dataclass(frozen=True)
class TexCoords:
    top_left: Vec2
    top_right: Vec2
    bottom_left: Vec2
    bottom_right: Vec2

    @classmethod
    def read(cls, reader: BinaryReader) -> TexCoords:
        return cls(reader.vec2(), reader.vec2(), reader.vec2(), reader.vec2())


@dataclass(frozen=True)
class Pane:
    name: str
    user_data: str
    flags: int
    base_position: int  # origin of the pane rectangle: x = value % 3 (left/center/right), y = value // 3
    alpha: int
    translate: Vec3
    rotate: Vec3  # degrees
    scale: Vec2
    size: Vec2
    children: tuple[Pane, ...]

    @property
    def visible(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def influenced_alpha(self) -> bool:
        """Whether this pane's alpha multiplies into its children's."""
        return bool(self.flags & 0x02)



@dataclass(frozen=True)
class Picture(Pane):
    vertex_colors: tuple[Color, Color, Color, Color]  # top left, top right, bottom left, bottom right
    material_index: int
    tex_coords: tuple[TexCoords, ...]


@dataclass(frozen=True)
class TextBox(Pane):
    text: str
    material_index: int
    font_index: int
    text_position: int  # where the text sits in the pane, like base_position
    text_alignment: int  # line alignment: 0 follows text_position, 1 left, 2 center, 3 right
    top_color: Color
    bottom_color: Color
    font_size: Vec2
    char_space: float
    line_space: float


@dataclass(frozen=True)
class WindowFrame:
    material_index: int
    flip: int  # 0 none, 1 horizontal, 2 vertical, 3 rotate 90, 4 rotate 180, 5 rotate 270


@dataclass(frozen=True)
class Window(Pane):
    inflation: tuple[float, float, float, float]  # left, right, top, bottom
    content_vertex_colors: tuple[Color, Color, Color, Color]
    content_material_index: int
    content_tex_coords: tuple[TexCoords, ...]
    frames: tuple[WindowFrame, ...]


@dataclass(frozen=True)
class Bounding(Pane):
    pass


@dataclass(frozen=True)
class Layout:
    canvas_size: Vec2
    textures: tuple[str, ...]
    fonts: tuple[str, ...]
    materials: tuple[Material, ...]
    root_pane: Optional[Pane]


@dataclass
class _Node:
    """A pane while its children are still being read."""

    kind: type
    fields: dict
    children: list = field(default_factory=list)

    def build(self):
        return self.kind(**self.fields, children=tuple(child.build() for child in self.children))


def read_brlyt(data: bytes) -> Layout:
    reader = BinaryReader(data)
    if reader.fourcc() != "RLYT":
        raise FormatError("Not a BRLYT layout.")
    reader.skip(4)  # byte order mark, version
    file_size = reader.u32()
    header_size = reader.u16()

    canvas_size = (0.0, 0.0)
    textures: tuple[str, ...] = ()
    fonts: tuple[str, ...] = ()
    materials: tuple[Material, ...] = ()
    root_pane: Optional[_Node] = None
    pane_stack: list[_Node] = []
    last_pane: Optional[_Node] = None

    position = header_size
    end = min(file_size, len(data))
    while position + 8 <= end:
        reader.seek(position)
        kind = reader.fourcc()
        size = reader.u32()
        if size < 8:
            break

        if kind == "lyt1":
            reader.skip(4)  # origin type
            canvas_size = reader.vec2()
        elif kind == "txl1":
            textures = _read_name_list(reader, position)
        elif kind == "fnl1":
            fonts = _read_name_list(reader, position)
        elif kind == "mat1":
            materials = _read_materials(reader, position)
        elif kind in _PANE_READERS:
            last_pane = _read_pane(reader, kind, position, size)
            if pane_stack:
                pane_stack[-1].children.append(last_pane)
            elif root_pane is None:
                root_pane = last_pane
        elif kind == "pas1" and last_pane is not None:
            pane_stack.append(last_pane)
        elif kind == "pae1" and pane_stack:
            pane_stack.pop()

        position += size

    return Layout(canvas_size, textures, fonts, materials, root_pane.build() if root_pane else None)


def _read_name_list(reader: BinaryReader, block: int) -> tuple[str, ...]:
    reader.seek(block + 8)
    count = reader.u16()
    reader.skip(2)
    table = block + 0x0C
    offsets = [reader.unpack("II")[0] for _ in range(count)]
    return tuple(reader.seek(table + offset).cstring() for offset in offsets)


def _read_materials(reader: BinaryReader, block: int) -> tuple[Material, ...]:
    reader.seek(block + 8)
    count = reader.u16()
    reader.skip(2)
    offsets = reader.unpack(f"{count}I")
    return tuple(Material.read(reader.seek(block + offset)) for offset in offsets)


def _read_pane_fields(reader: BinaryReader) -> dict:
    flags, base_position, alpha, _ = reader.unpack("4B")
    return dict(flags=flags, base_position=base_position, alpha=alpha,
                name=reader.fixed_string(16), user_data=reader.fixed_string(8),
                translate=reader.vec3(), rotate=reader.vec3(), scale=reader.vec2(), size=reader.vec2())


def _read_picture(reader: BinaryReader, block: int, size: int) -> dict:
    vertex_colors = tuple(reader.color() for _ in range(4))
    material_index, coord_count = reader.unpack("HBx")
    return dict(vertex_colors=vertex_colors, material_index=material_index,
                tex_coords=tuple(TexCoords.read(reader) for _ in range(coord_count)))


def _read_text_box(reader: BinaryReader, block: int, size: int) -> dict:
    _, text_length, material_index, font_index, text_position, text_alignment = reader.unpack("4H2Bxx")
    text_offset = reader.u32()
    top_color, bottom_color = reader.color(), reader.color()
    font_size = reader.vec2()
    char_space, line_space = reader.unpack("2f")
    start = block + text_offset
    raw = bytes(reader.data[start:min(start + text_length, block + size)])
    text = raw[:len(raw) & ~1].decode("utf-16-be", errors="replace").rstrip("\0")
    return dict(text=text, material_index=material_index, font_index=font_index,
                text_position=text_position, text_alignment=text_alignment, top_color=top_color,
                bottom_color=bottom_color, font_size=font_size, char_space=char_space, line_space=line_space)


def _read_window(reader: BinaryReader, block: int, size: int) -> dict:
    inflation = reader.unpack("4f")
    frame_count = reader.u8()
    reader.skip(3)
    content_offset, frame_table_offset = reader.unpack("II")

    reader.seek(block + content_offset)
    content_vertex_colors = tuple(reader.color() for _ in range(4))
    content_material_index, coord_count = reader.unpack("HBx")
    content_tex_coords = tuple(TexCoords.read(reader) for _ in range(coord_count))

    reader.seek(block + frame_table_offset)
    frame_offsets = reader.unpack(f"{frame_count}I")
    frames = tuple(WindowFrame(*reader.seek(block + offset).unpack("HBx")) for offset in frame_offsets)
    return dict(inflation=inflation, content_vertex_colors=content_vertex_colors,
                content_material_index=content_material_index, content_tex_coords=content_tex_coords,
                frames=frames)


_PANE_READERS = {
    "pan1": (Pane, None),
    "bnd1": (Bounding, None),
    "pic1": (Picture, _read_picture),
    "txt1": (TextBox, _read_text_box),
    "wnd1": (Window, _read_window),
}


def _read_pane(reader: BinaryReader, kind: str, block: int, size: int) -> _Node:
    pane_type, read_content = _PANE_READERS[kind]
    reader.seek(block + 8)
    fields = _read_pane_fields(reader)
    if read_content is not None:
        fields.update(read_content(reader, block, size))
    return _Node(pane_type, fields)

