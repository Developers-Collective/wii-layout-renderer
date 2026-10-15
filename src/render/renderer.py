"""Drawing render states with QPainter.

Panes are drawn in tree order. Every quad (a picture, a window piece, a text run) is shaded once into a
cached image; per frame, only those images are mapped onto the screen with each pane's transform.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace
from typing import Callable, Optional, Sequence

import numpy as np

from PyQt6 import QtCore, QtGui

from ..formats import Brfnt
from ..resources import Resources, Texture
from ..state import (Color, LayoutState, MaterialState, PaneState, PictureState, TexCoords, TextBoxState,
                     WindowState)
from . import window
from .geometry import IDENTITY, Matrix, base_point, multiply, pane_matrix, plane_affine
from .shading import (FULL_UV, Combiner, TextureLayer, quad_resolution, shade_quad, shade_text,
                      texture_matrix)
from .text import TextRun, layout_text, line_alignment

DEFAULT_CANVAS_SIZE = (608.0, 456.0)
FRAME_VERTEX_COLORS = tuple(Color() for _ in range(4))  # window frames are not tinted

_ARGB32_PREMULTIPLIED = QtGui.QImage.Format.Format_ARGB32_Premultiplied
_SMOOTH = QtGui.QPainter.RenderHint.SmoothPixmapTransform


class LayoutRenderer:
    """Draws ``LayoutState``s with the textures and fonts of a ``Resources``."""

    def __init__(self, resources: Optional[Resources] = None, *, cache_size: int = 2048):
        self._resources = resources or Resources()
        self._cache: OrderedDict[tuple, object] = OrderedDict()
        self._cache_size = cache_size

    @property
    def resources(self) -> Resources:
        return self._resources

    @resources.setter
    def resources(self, resources: Resources) -> None:
        if resources is not self._resources:
            self._resources = resources
            self.clear_cache()

    def clear_cache(self) -> None:
        self._cache.clear()

    def paint(self, painter: QtGui.QPainter, state: LayoutState, target: QtCore.QRectF,
              background: Optional[QtGui.QColor] = None) -> None:
        """Draw the state's canvas into ``target``, stretched to fill it, and nothing outside it."""
        canvas_width, canvas_height = canvas_size(state)
        painter.save()
        try:
            painter.setClipRect(target)
            if background is not None:
                painter.fillRect(target, background)
            scale_x = target.width() / canvas_width
            scale_y = target.height() / canvas_height
            view = QtGui.QTransform(scale_x, 0.0, 0.0, -scale_y, target.center().x(), target.center().y())
            if state.root is not None:
                _StatePainter(self, painter, state, view).draw_pane(state.root, IDENTITY, 1.0)
        finally:
            painter.restore()

    def render_image(self, state: LayoutState, width: int, height: int,
                     background: Optional[QtGui.QColor] = None) -> QtGui.QImage:
        """The state's canvas as a ``width`` x ``height`` image."""
        image = QtGui.QImage(width, height, _ARGB32_PREMULTIPLIED)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        try:
            self.paint(painter, state, QtCore.QRectF(0, 0, width, height), background)
        finally:
            painter.end()
        return image

    def cached(self, key: tuple, make: Callable[[], object]) -> object:
        try:
            self._cache.move_to_end(key)
            return self._cache[key]
        except KeyError:
            pass
        value = make()
        self._cache[key] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value


def canvas_size(state: LayoutState) -> tuple[float, float]:
    width, height = state.canvas_size.x, state.canvas_size.y
    return (width if width > 0 else DEFAULT_CANVAS_SIZE[0]), (height if height > 0 else DEFAULT_CANVAS_SIZE[1])


class _StatePainter:
    """Draws one state with one painter."""

    def __init__(self, renderer: LayoutRenderer, painter: QtGui.QPainter, state: LayoutState,
                 view: QtGui.QTransform):
        self.renderer = renderer
        self.resources = renderer.resources
        self.painter = painter
        self.state = state
        self.view = view

    def draw_pane(self, pane: PaneState, parent_matrix: Matrix, parent_alpha: float) -> None:
        if not pane.visible:
            return  # hides the children too
        matrix = multiply(parent_matrix, pane_matrix(pane))
        alpha = parent_alpha * pane.alpha / 255.0
        world = QtGui.QTransform(*plane_affine(matrix)) * self.view

        if isinstance(pane, PictureState):
            x, y = base_point(pane)
            self.draw_quad(world, x, y, pane.size.x, pane.size.y, pane.material_index, pane.vertex_colors,
                           pane.tex_coords, alpha)
        elif isinstance(pane, WindowState):
            self.draw_window(world, pane, alpha)
        elif isinstance(pane, TextBoxState):
            self.draw_text(world, pane, alpha)

        child_alpha = alpha if pane.influenced_alpha else parent_alpha
        for child in pane.children:
            self.draw_pane(child, matrix, child_alpha)

    # Quads

    def draw_quad(self, world: QtGui.QTransform, x: float, y: float, width: float, height: float,
                  material_index: int, vertex_colors: Sequence, tex_coords: Sequence[TexCoords],
                  alpha: float) -> None:
        if width == 0 or height == 0 or alpha <= 0:
            return
        material = self.state.material(material_index)
        colors = tuple(color.as_tuple() for color in vertex_colors)
        screen_s, screen_t = _screen_lengths(world, x, y, width, height)
        layers = [self.texture_layer(material, slot, tex_coords, screen_s, screen_t) for slot in (0, 1)]
        present = [layer for layer in layers if layer is not None]
        combiner = _combiner(material)
        resolution = quad_resolution(present, len(set(colors)) == 1)

        key = ("quad", resolution, colors, tuple(_layer_key(layer) for layer in layers), combiner)
        image = self.renderer.cached(key, lambda: _to_image(shade_quad(*resolution, colors, layers, combiner)))
        smooth = present[0].linear if present else True
        self.draw_image(image, world, x, y, width, height, 1.0 if combiner.opaque else alpha, smooth)

    def texture_layer(self, material: Optional[MaterialState], slot: int, tex_coords: Sequence[TexCoords],
                      screen_s: float, screen_t: float) -> Optional[TextureLayer]:
        if material is None or slot >= len(material.texture_maps):
            return None
        texture_map = material.texture_maps[slot]
        texture = self.resources.texture(texture_map.texture) if texture_map.texture else None
        if texture is None:
            return None
        coords = tex_coords[slot] if slot < len(tex_coords) else (tex_coords[0] if tex_coords else None)
        uv = _uv_corners(coords) if coords is not None else FULL_UV
        srt = material.texture_srts[slot] if slot < len(material.texture_srts) else None
        layer = TextureLayer(texture, uv, texture_matrix(srt), texture_map.wrap_s, texture_map.wrap_t, True)
        # GX filters with the magnification filter where a texel covers more than a pixel.
        span_s, span_t = layer.texel_spans()
        magnified = screen_s > span_s or screen_t > span_t
        return replace(layer, linear=texture_map.mag_linear if magnified else texture_map.min_linear)

    def draw_image(self, image: QtGui.QImage, world: QtGui.QTransform, x: float, y: float, width: float,
                   height: float, opacity: float, smooth: bool) -> None:
        """Map the image onto the rectangle with its top left corner at (x, y) in pane coordinates."""
        to_pane = QtGui.QTransform(width / image.width(), 0.0, 0.0, -height / image.height(), x, y)
        self.painter.setTransform(to_pane * world)
        self.painter.setOpacity(opacity)
        self.painter.setRenderHint(_SMOOTH, smooth)
        self.painter.drawImage(QtCore.QPointF(0.0, 0.0), image)

    # Windows

    def draw_window(self, world: QtGui.QTransform, pane: WindowState, alpha: float) -> None:
        base = base_point(pane)
        frame_count = len(pane.frames)
        frame = window.frame_size(frame_count, lambda index: self.frame_texture_size(pane, index))

        content = window.content_quad(base, pane.size, frame, pane.inflation)
        self.draw_quad(world, *content, pane.content_material_index, pane.content_vertex_colors,
                       pane.content_tex_coords, alpha)

        for piece in window.frame_pieces(frame_count, base, pane.size, frame):
            texture = self.frame_texture(pane, piece.frame)
            quad = piece.quad
            if texture is None or quad.width <= 0 or quad.height <= 0:
                continue
            flip = piece.flip if piece.flip is not None else pane.frames[piece.frame].flip
            uv = window.frame_tex_coords(piece.anchor, quad.width, quad.height, texture.width, texture.height, flip)
            self.draw_quad(world, *quad, pane.frames[piece.frame].material_index, FRAME_VERTEX_COLORS, [uv], alpha)

    def frame_texture(self, pane: WindowState, index: int) -> Optional[Texture]:
        material = self.state.material(pane.frames[index].material_index) if index < len(pane.frames) else None
        if material is None or not material.texture_maps or not material.texture_maps[0].texture:
            return None
        return self.resources.texture(material.texture_maps[0].texture)

    def frame_texture_size(self, pane: WindowState, index: int) -> tuple[float, float]:
        texture = self.frame_texture(pane, index)
        return (texture.width, texture.height) if texture is not None else (0.0, 0.0)

    # Text

    def draw_text(self, world: QtGui.QTransform, pane: TextBoxState, alpha: float) -> None:
        if not pane.text or pane.size.x == 0 or pane.size.y == 0 or alpha <= 0:
            return
        font = self.resources.font(pane.font) if pane.font else None
        run_key, run = self.text_run(pane, font)
        if run is None or run.width <= 0 or run.height <= 0:
            return

        x, y = base_point(pane)
        horizontal, vertical = pane.text_position % 3, min(pane.text_position // 3, 2)
        x += (0.0, (pane.size.x - run.width) / 2, pane.size.x - run.width)[horizontal]
        y -= (0.0, (pane.size.y - run.height) / 2, pane.size.y - run.height)[vertical]

        top, bottom = pane.top_color.as_tuple(), pane.bottom_color.as_tuple()
        combiner = _combiner(self.state.material(pane.material_index))
        key = ("text", run_key, top, bottom, combiner)
        image = self.renderer.cached(key, lambda: _to_image(shade_text(run.coverage, top, bottom, combiner)))
        self.draw_image(image, world, x + run.left, y, run.image_width, run.image_height, alpha, True)

    def text_run(self, pane: TextBoxState, font: Optional[Brfnt]) -> tuple[tuple, Optional[TextRun]]:
        """The pane's text laid out, and the cache key it is stored under."""
        alignment = line_alignment(pane.text_alignment, pane.text_position)
        if font is not None:
            font_size = (pane.font_size.x, pane.font_size.y)
            key = ("run", pane.text, id(font), font_size, pane.char_space, pane.line_space, alignment)
            return key, self.renderer.cached(key, lambda: layout_text(
                pane.text, font, font_size, pane.char_space, pane.line_space, alignment))
        # Without the font, a system font stands in, rendered at about the size it appears on screen.
        pixel_height = max(8, round(abs(pane.font_size.y * self.view.m22())))
        key = ("system run", pane.text, pane.font_size.y, pixel_height, alignment)
        return key, self.renderer.cached(key, lambda: _system_text_run(
            pane.text, pane.font_size.y, pixel_height, alignment))


def _uv_corners(coords: TexCoords) -> tuple:
    return tuple((corner.x, corner.y) for corner in
                 (coords.top_left, coords.top_right, coords.bottom_left, coords.bottom_right))


def _combiner(material: Optional[MaterialState]) -> Combiner:
    if material is None:
        return Combiner()
    return Combiner(material.black_color.as_tuple(), material.white_color.as_tuple(), material.alpha_compare,
                    material.blend_mode is not None and material.blend_mode.type == 0)


def _layer_key(layer: Optional[TextureLayer]) -> Optional[tuple]:
    if layer is None:
        return None
    return id(layer.texture), layer.uv, layer.matrix, layer.wrap_s, layer.wrap_t, layer.linear


def _screen_lengths(world: QtGui.QTransform, x: float, y: float, width: float, height: float) -> tuple[float, float]:
    """How long the quad's top and left edges are on screen, in pixels."""
    origin = world.map(QtCore.QPointF(x, y))
    across = world.map(QtCore.QPointF(x + width, y)) - origin
    down = world.map(QtCore.QPointF(x, y - height)) - origin
    return math.hypot(across.x(), across.y()), math.hypot(down.x(), down.y())


def _to_image(argb: np.ndarray) -> QtGui.QImage:
    height, width = argb.shape
    data = np.ascontiguousarray(argb, dtype=np.uint32).tobytes()
    return QtGui.QImage(data, width, height, width * 4, _ARGB32_PREMULTIPLIED).copy()


def _system_text_run(text: str, font_height: float, pixel_height: int, alignment: int) -> Optional[TextRun]:
    """Glyph coverage drawn with the default system font, for text boxes whose BRFNT is not available."""
    font = QtGui.QFont()
    font.setPixelSize(pixel_height)
    metrics = QtGui.QFontMetricsF(font)
    lines = text.replace("\r", "").split("\n")
    width = math.ceil(max(metrics.horizontalAdvance(line) for line in lines)) + 2
    height = math.ceil(metrics.height() * len(lines)) + 2
    if width <= 2 or font_height <= 0:
        return None

    image = QtGui.QImage(width, height, _ARGB32_PREMULTIPLIED)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    try:
        painter.setFont(font)
        painter.setPen(QtGui.QColor(255, 255, 255))
        option = QtGui.QTextOption((QtCore.Qt.AlignmentFlag.AlignLeft, QtCore.Qt.AlignmentFlag.AlignHCenter,
                                    QtCore.Qt.AlignmentFlag.AlignRight)[alignment])
        painter.drawText(QtCore.QRectF(1, 1, width - 2, height - 2), text, option)
    finally:
        painter.end()

    pixels = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), dtype=np.uint32)
    alpha = (pixels.reshape(height, image.bytesPerLine() // 4)[:, :width] >> 24).astype(np.uint8)
    coverage = np.stack((np.full_like(alpha, 255), alpha), axis=-1)
    scale = font_height * len(lines) / height
    return TextRun(coverage, scale, scale, width * scale, height * scale)
