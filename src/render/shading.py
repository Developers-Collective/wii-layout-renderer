"""What a material makes of a quad's pixels, computed with numpy.

Modelled: texture 0 through its texture coordinates and SRT, times the vertex colours; the TEV colour
registers remapping texture black and white; texture 1 masking the alpha; the alpha compare; blend mode
"none". A quad is shaded in its own s/t space at about one pixel per texel, so moving or scaling a pane
never shades it again. Results are premultiplied 0xAARRGGBB uint32 arrays (QImage ARGB32_Premultiplied).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..formats.brlyt import AlphaCompare
from ..resources import Texture
from ..state import TextureSrt

MAX_RESOLUTION = 2048
GRADIENT_RESOLUTION = 32  # pixels per axis for vertex colour gradients across sparse textures

ColorTuple = tuple[int, int, int, int]
UvCorners = tuple[tuple[float, float], ...]  # top left, top right, bottom left, bottom right
TexMatrix = tuple[float, float, float, float, float, float]  # m00, m01, m03, m10, m11, m13

IDENTITY_TEX_MATRIX: TexMatrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
FULL_UV: UvCorners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


@dataclass(frozen=True)
class TextureLayer:
    """A texture as one quad samples it."""

    texture: Texture
    uv: UvCorners
    matrix: TexMatrix
    wrap_s: int  # 0 clamp, 1 repeat, 2 mirror
    wrap_t: int
    linear: bool

    def texel_spans(self) -> tuple[float, float]:
        """How many texels the quad covers along s and along t."""
        width, height = self.texture.width, self.texture.height
        tl, tr, bl, br = ((u * width, v * height) for u, v in (_apply(self.matrix, uv) for uv in self.uv))
        return max(math.dist(tl, tr), math.dist(bl, br)), max(math.dist(tl, bl), math.dist(tr, br))


@dataclass(frozen=True)
class Combiner:
    """The material settings that combine the colours of a quad."""

    black: ColorTuple = (0, 0, 0, 0)
    white: ColorTuple = (255, 255, 255, 255)
    alpha_compare: Optional[AlphaCompare] = None
    opaque: bool = False  # blend mode "none"


def texture_matrix(srt: Optional[TextureSrt]) -> TexMatrix:
    """The texture coordinate matrix of an SRT, which scales and rotates about the texture center."""
    if srt is None:
        return IDENTITY_TEX_MATRIX
    angle = math.radians(srt.rotate)
    cos, sin = math.cos(angle), math.sin(angle)
    m00, m01 = cos * srt.scale.x, -sin * srt.scale.y
    m10, m11 = sin * srt.scale.x, cos * srt.scale.y
    return (m00, m01, srt.translate.x + 0.5 - 0.5 * (m00 + m01),
            m10, m11, srt.translate.y + 0.5 - 0.5 * (m10 + m11))


def quad_resolution(layers: Sequence[TextureLayer], uniform_colors: bool) -> tuple[int, int]:
    """The image size to shade a quad at: about one pixel per texel of its densest texture."""
    if not layers:
        size = 1 if uniform_colors else GRADIENT_RESOLUTION
        return size, size
    spans = [layer.texel_spans() for layer in layers]
    width = max(span_s for span_s, _ in spans)
    height = max(span_t for _, span_t in spans)
    minimum = 1 if uniform_colors else GRADIENT_RESOLUTION
    return (min(MAX_RESOLUTION, max(minimum, math.ceil(width - 1e-4))),
            min(MAX_RESOLUTION, max(minimum, math.ceil(height - 1e-4))))


def shade_quad(width: int, height: int, vertex_colors: Sequence[ColorTuple],
               layers: Sequence[Optional[TextureLayer]], combiner: Combiner) -> np.ndarray:
    """Shade a quad into a (height, width) premultiplied ARGB image. ``layers`` holds the material's
    texture 0 and texture 1, each None when the material has none."""
    s = ((np.arange(width, dtype=np.float32) + 0.5) / width)[None, :]
    t = ((np.arange(height, dtype=np.float32) + 0.5) / height)[:, None]
    vertex = _bilinear(np.asarray(vertex_colors, dtype=np.float32), s, t)

    first = layers[0] if layers else None
    texel = _sample_layer(first, s, t) if first is not None else np.full(4, 255.0, dtype=np.float32)
    color = _remap(texel, combiner) * vertex / 255.0

    second = layers[1] if len(layers) > 1 else None
    if second is not None:
        mask = _sample_layer(second, s, t)
        color[..., 3] *= (mask[..., 3] if second.texture.has_alpha else mask[..., 0]) / 255.0

    alpha = np.floor(color[..., 3])
    passed = _alpha_passes(combiner.alpha_compare, alpha)
    alpha = np.where(passed, 255.0 if combiner.opaque else alpha, 0.0)
    return _premultiplied_argb(color[..., :3], alpha)


def shade_text(texels: np.ndarray, top: ColorTuple, bottom: ColorTuple, combiner: Combiner) -> np.ndarray:
    """Colour a text run's RGBA glyph texels into a premultiplied ARGB image, like a texture: the material
    remaps them and the text colour, fading from ``top`` to ``bottom``, multiplies them. Pixels no glyph
    covers stay empty."""
    height = texels.shape[0]
    glyph = _remap(texels.astype(np.float32), combiner)

    t = ((np.arange(height, dtype=np.float32) + 0.5) / height)[:, None, None]
    top_color, bottom_color = np.asarray(top, np.float32), np.asarray(bottom, np.float32)
    gradient = np.floor(top_color + (bottom_color - top_color) * t)
    alpha = np.where(texels[..., 3] > 0, glyph[..., 3] * gradient[..., 3] / 255.0, 0.0)
    return _premultiplied_argb(gradient[..., :3] * glyph[..., :3] / 255.0, alpha)


def _premultiplied_argb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 255.0)
    channels = np.clip(rgb, 0.0, 255.0) * (alpha / 255.0)[..., None]
    a = alpha.astype(np.uint32)
    r, g, b = (np.rint(channels[..., index]).astype(np.uint32) for index in range(3))
    return (a << 24) | (r << 16) | (g << 8) | b


def _apply(matrix: TexMatrix, uv: tuple[float, float]) -> tuple[float, float]:
    m00, m01, m03, m10, m11, m13 = matrix
    u, v = uv
    return m00 * u + m01 * v + m03, m10 * u + m11 * v + m13


def _bilinear(corners: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Blend values at the top left, top right, bottom left and bottom right corners across s and t."""
    s, t = s[..., None], t[..., None]
    top = corners[0] + (corners[1] - corners[0]) * s
    bottom = corners[2] + (corners[3] - corners[2]) * s
    return top + (bottom - top) * t


def _sample_layer(layer: TextureLayer, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    uv = _bilinear(np.asarray(layer.uv, dtype=np.float32), s, t)
    m00, m01, m03, m10, m11, m13 = layer.matrix
    u = m00 * uv[..., 0] + m01 * uv[..., 1] + m03
    v = m10 * uv[..., 0] + m11 * uv[..., 1] + m13
    return _sample(layer.texture.pixels, u, v, layer.wrap_s, layer.wrap_t, layer.linear)


def _sample(pixels: np.ndarray, u: np.ndarray, v: np.ndarray, wrap_s: int, wrap_t: int, linear: bool) -> np.ndarray:
    """Texture colours at normalized coordinates, as float32 RGBA. Texel centers sit at half texels."""
    height, width = pixels.shape[:2]
    x = u * width - 0.5
    y = v * height - 0.5
    if not linear:
        columns = _wrap(np.floor(x + 0.5).astype(np.int64), width, wrap_s)
        rows = _wrap(np.floor(y + 0.5).astype(np.int64), height, wrap_t)
        return pixels[rows, columns].astype(np.float32)

    x0, y0 = np.floor(x), np.floor(y)
    dx, dy = (x - x0)[..., None], (y - y0)[..., None]
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    left, right = _wrap(x0, width, wrap_s), _wrap(x0 + 1, width, wrap_s)
    upper, lower = _wrap(y0, height, wrap_t), _wrap(y0 + 1, height, wrap_t)
    top_left = pixels[upper, left].astype(np.float32)
    bottom_left = pixels[lower, left].astype(np.float32)
    top = top_left + (pixels[upper, right] - top_left) * dx
    bottom = bottom_left + (pixels[lower, right] - bottom_left) * dx
    return top + (bottom - top) * dy


def _wrap(index: np.ndarray, size: int, mode: int) -> np.ndarray:
    if mode == 1:  # repeat
        return np.mod(index, size)
    if mode == 2:  # mirror
        period = np.mod(index, 2 * size)
        return np.where(period < size, period, 2 * size - 1 - period)
    return np.clip(index, 0, size - 1)


def _remap(texel: np.ndarray, combiner: Combiner) -> np.ndarray:
    """Map texel 0 to the black colour and 255 to the white colour, per channel."""
    black = np.asarray(combiner.black, dtype=np.float32)
    white = np.asarray(combiner.white, dtype=np.float32)
    return np.clip(black + (white - black) * texel / 255.0, 0.0, 255.0)


_COMPARE = (
    lambda a, ref: np.zeros_like(a, dtype=bool),  # never
    lambda a, ref: a < ref,
    lambda a, ref: a == ref,
    lambda a, ref: a <= ref,
    lambda a, ref: a > ref,
    lambda a, ref: a != ref,
    lambda a, ref: a >= ref,
    lambda a, ref: np.ones_like(a, dtype=bool),  # always
)


def _alpha_passes(test: Optional[AlphaCompare], alpha: np.ndarray) -> np.ndarray:
    if test is None:
        return np.ones_like(alpha, dtype=bool)
    first = _COMPARE[test.comp0 & 7](alpha, test.ref0)
    second = _COMPARE[test.comp1 & 7](alpha, test.ref1)
    op = test.op & 3
    if op == 0:
        return first & second
    if op == 1:
        return first | second
    if op == 2:
        return first ^ second
    return ~(first ^ second)
