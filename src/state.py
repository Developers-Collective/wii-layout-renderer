"""The render state: a mutable snapshot of everything the renderer reads to draw one frame.

Every frame starts from a fresh copy of the layout file; the animations are applied to it, then the state
hooks may change anything in it before it is drawn. Coordinates are layout units with the origin at the
canvas center and Y pointing up; angles are degrees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, TypeVar, Union

from .formats import brlyt


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Color:
    """An RGBA colour. Channels are 0-255; TEV colours may also hold the signed 10-bit range."""

    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.r, self.g, self.b, self.a


@dataclass
class TexCoords:
    """The texture coordinates at a quad's four corners."""

    top_left: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    top_right: Vec2 = field(default_factory=lambda: Vec2(1.0, 0.0))
    bottom_left: Vec2 = field(default_factory=lambda: Vec2(0.0, 1.0))
    bottom_right: Vec2 = field(default_factory=lambda: Vec2(1.0, 1.0))


@dataclass
class TextureMap:
    texture: Optional[str] = None  # texture file name, None when the material points nowhere
    wrap_s: int = 0  # 0 clamp, 1 repeat, 2 mirror
    wrap_t: int = 0
    min_linear: bool = True
    mag_linear: bool = True


@dataclass
class TextureSrt:
    translate: Vec2 = field(default_factory=Vec2)
    rotate: float = 0.0
    scale: Vec2 = field(default_factory=lambda: Vec2(1.0, 1.0))


@dataclass
class MaterialState:
    name: str = ""
    # TEV colour registers. Textures are remapped so 0 becomes ``black_color`` and 255 ``white_color``.
    tev_colors: list[Color] = field(default_factory=lambda: [Color(0, 0, 0, 0), Color(), Color()])
    texture_maps: list[TextureMap] = field(default_factory=list)
    texture_srts: list[TextureSrt] = field(default_factory=list)
    alpha_compare: Optional[brlyt.AlphaCompare] = None
    blend_mode: Optional[brlyt.BlendMode] = None

    @property
    def black_color(self) -> Color:
        return self.tev_colors[0]

    @property
    def white_color(self) -> Color:
        return self.tev_colors[1]


@dataclass(eq=False)
class PaneState:
    name: str = ""
    user_data: str = ""
    visible: bool = True
    influenced_alpha: bool = False  # whether this pane's alpha multiplies into its children's
    alpha: int = 255
    base_position: int = 4  # origin of the pane rectangle: x = value % 3 (left/center/right), y = value // 3
    translate: Vec3 = field(default_factory=Vec3)
    rotate: Vec3 = field(default_factory=Vec3)
    scale: Vec2 = field(default_factory=lambda: Vec2(1.0, 1.0))
    size: Vec2 = field(default_factory=Vec2)
    children: list[PaneState] = field(default_factory=list)

    def walk(self) -> Iterator[PaneState]:
        """This pane and every pane below it, parents before children (the drawing order)."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, name: str) -> Optional[PaneState]:
        return next((pane for pane in self.walk() if pane.name == name), None)


@dataclass(eq=False)
class PictureState(PaneState):
    vertex_colors: list[Color] = field(default_factory=lambda: [Color() for _ in range(4)])  # TL, TR, BL, BR
    material_index: int = 0
    tex_coords: list[TexCoords] = field(default_factory=list)


@dataclass(eq=False)
class TextBoxState(PaneState):
    text: str = ""
    font: Optional[str] = None  # font file name
    font_size: Vec2 = field(default_factory=Vec2)
    char_space: float = 0.0
    line_space: float = 0.0
    text_position: int = 4  # where the text sits in the pane, like base_position
    text_alignment: int = 0  # line alignment: 0 follows text_position, 1 left, 2 center, 3 right
    top_color: Color = field(default_factory=Color)
    bottom_color: Color = field(default_factory=Color)
    material_index: int = 0


@dataclass
class WindowFrameState:
    material_index: int = 0
    flip: int = 0  # 0 none, 1 horizontal, 2 vertical, 3 rotate 90, 4 rotate 180, 5 rotate 270


@dataclass
class WindowInflation:
    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0


@dataclass(eq=False)
class WindowState(PaneState):
    inflation: WindowInflation = field(default_factory=WindowInflation)
    content_vertex_colors: list[Color] = field(default_factory=lambda: [Color() for _ in range(4)])
    content_material_index: int = 0
    content_tex_coords: list[TexCoords] = field(default_factory=list)
    frames: list[WindowFrameState] = field(default_factory=list)


@dataclass(eq=False)
class BoundingState(PaneState):
    """A bounding pane, which is never drawn."""


PaneT = TypeVar("PaneT", bound=PaneState)


@dataclass(eq=False)
class LayoutState:
    canvas_size: Vec2 = field(default_factory=lambda: Vec2(608.0, 456.0))
    root: Optional[PaneState] = None
    materials: list[MaterialState] = field(default_factory=list)

    @classmethod
    def from_layout(cls, layout: brlyt.Layout) -> LayoutState:
        """A fresh snapshot of a layout as stored in its file."""
        return cls(Vec2(*layout.canvas_size),
                   _pane_state(layout.root_pane, layout) if layout.root_pane is not None else None,
                   [_material_state(material, layout) for material in layout.materials])

    def panes(self) -> Iterator[PaneState]:
        """Every pane, parents before children (the drawing order)."""
        if self.root is not None:
            yield from self.root.walk()

    def panes_by_name(self) -> dict[str, PaneState]:
        """Every pane by name, the first one for names used twice. Build it once and reuse it within a hook."""
        panes: dict[str, PaneState] = {}
        for pane in self.panes():
            panes.setdefault(pane.name, pane)
        return panes

    def find_pane(self, name: str, kind: type[PaneT] = PaneState) -> Optional[PaneT]:
        """The first pane with this name, or None. ``kind`` narrows the type, e.g. ``TextBoxState``."""
        pane = self.root.find(name) if self.root is not None else None
        return pane if isinstance(pane, kind) else None

    def pane(self, name: str, kind: type[PaneT] = PaneState) -> PaneT:
        """Like ``find_pane`` but raises KeyError when there is no such pane."""
        pane = self.find_pane(name, kind)
        if pane is None:
            raise KeyError(f"The layout has no {kind.__name__} named {name!r}.")
        return pane

    def material(self, index_or_name: Union[int, str]) -> Optional[MaterialState]:
        if isinstance(index_or_name, int):
            return self.materials[index_or_name] if 0 <= index_or_name < len(self.materials) else None
        return next((material for material in self.materials if material.name == index_or_name), None)


def _tex_coords(coords: brlyt.TexCoords) -> TexCoords:
    return TexCoords(Vec2(*coords.top_left), Vec2(*coords.top_right),
                     Vec2(*coords.bottom_left), Vec2(*coords.bottom_right))


def _material_state(material: brlyt.Material, layout: brlyt.Layout) -> MaterialState:
    def texture_name(index: int) -> Optional[str]:
        return layout.textures[index] if 0 <= index < len(layout.textures) else None

    return MaterialState(
        name=material.name,
        tev_colors=[Color(*color) for color in material.tev_colors],
        texture_maps=[TextureMap(texture_name(tex_map.texture_index), tex_map.wrap_s, tex_map.wrap_t,
                                 tex_map.min_linear, tex_map.mag_linear) for tex_map in material.tex_maps],
        texture_srts=[TextureSrt(Vec2(*srt.translate), srt.rotate, Vec2(*srt.scale)) for srt in material.tex_srts],
        alpha_compare=material.alpha_compare,
        blend_mode=material.blend_mode,
    )


def _pane_state(pane: brlyt.Pane, layout: brlyt.Layout) -> PaneState:
    common = dict(
        name=pane.name, user_data=pane.user_data, visible=pane.visible, influenced_alpha=pane.influenced_alpha,
        alpha=pane.alpha, base_position=pane.base_position,
        translate=Vec3(*pane.translate), rotate=Vec3(*pane.rotate), scale=Vec2(*pane.scale),
        size=Vec2(*pane.size), children=[_pane_state(child, layout) for child in pane.children],
    )
    if isinstance(pane, brlyt.Picture):
        return PictureState(**common, vertex_colors=[Color(*color) for color in pane.vertex_colors],
                            material_index=pane.material_index,
                            tex_coords=[_tex_coords(coords) for coords in pane.tex_coords])
    if isinstance(pane, brlyt.TextBox):
        font = layout.fonts[pane.font_index] if pane.font_index < len(layout.fonts) else None
        return TextBoxState(**common, text=pane.text, font=font, font_size=Vec2(*pane.font_size),
                            char_space=pane.char_space, line_space=pane.line_space,
                            text_position=pane.text_position, text_alignment=pane.text_alignment,
                            top_color=Color(*pane.top_color), bottom_color=Color(*pane.bottom_color),
                            material_index=pane.material_index)
    if isinstance(pane, brlyt.Window):
        return WindowState(**common, inflation=WindowInflation(*pane.inflation),
                           content_vertex_colors=[Color(*color) for color in pane.content_vertex_colors],
                           content_material_index=pane.content_material_index,
                           content_tex_coords=[_tex_coords(coords) for coords in pane.content_tex_coords],
                           frames=[WindowFrameState(frame.material_index, frame.flip) for frame in pane.frames])
    if isinstance(pane, brlyt.Bounding):
        return BoundingState(**common)
    return PaneState(**common)
