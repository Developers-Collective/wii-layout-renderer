"""How NW4R draws a window pane: a content quad inside a frame made of 1, 4 or 8 pieces.

With one frame, its corner texture is drawn at all four corners, flipped, each piece running along an
edge. With four, each corner has its own frame and flip. With eight, the corners and the edges are separate
pieces. A piece's texture keeps its size from the corner it is anchored at; past the texture, the material's
wrap mode decides what is drawn (clamping stretches the texture's outer row or column).
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

from ..state import TexCoords, Vec2, WindowInflation

# Frame indices, in the order a window stores its frames.
LT, RT, LB, RB, L, R, T, B = range(8)

# Flips: none, horizontal, vertical, rotate 90, rotate 180, rotate 270.
FLIP_NONE, FLIP_H, FLIP_V, FLIP_90, FLIP_180, FLIP_270 = range(6)

# For each flip, the texture coordinates at the texture's top left, top right, bottom left and bottom right
# corners, and which coordinate runs along x and which along y.
_FLIPS = (
    ((0, 0), (1, 0), (0, 1), (1, 1), 0, 1),
    ((1, 0), (0, 0), (1, 1), (0, 1), 0, 1),
    ((0, 1), (1, 1), (0, 0), (1, 0), 0, 1),
    ((0, 1), (0, 0), (1, 1), (1, 0), 1, 0),
    ((1, 1), (0, 1), (1, 0), (0, 0), 0, 1),
    ((1, 0), (1, 1), (0, 0), (0, 1), 1, 0),
)

# For each corner a piece is anchored at: the corner of the flip table it starts from, the corners it
# extends towards along x and y, and which of the quad's corners (0 top left, 1 top right, 2 bottom left,
# 3 bottom right) keep the start coordinate and which take the extended one, along x and along y.
_ANCHORS = {
    LT: (0, 1, 2, (0, 2), (0, 1), (3, 1), (3, 2)),
    RT: (1, 0, 3, (1, 3), (1, 0), (2, 0), (2, 3)),
    LB: (2, 3, 0, (2, 0), (2, 3), (1, 3), (1, 0)),
    RB: (3, 2, 1, (3, 1), (3, 2), (0, 2), (0, 1)),
}


class FrameSize(NamedTuple):
    left: float
    right: float
    top: float
    bottom: float


class Quad(NamedTuple):
    """A rectangle in pane coordinates: its top left corner and size (Y up, so it extends downwards)."""

    x: float
    y: float
    width: float
    height: float


class FramePiece(NamedTuple):
    frame: int  # which of the window's frames draws the piece
    anchor: int  # the corner (LT, RT, LB, RB) whose texture placement the piece uses
    flip: Optional[int]  # the flip to draw with, or None for the frame's own
    quad: Quad


def frame_size(frame_count: int, texture_size: Callable[[int], tuple[float, float]]) -> FrameSize:
    """How far the frame reaches in from each edge: the size of the corner textures. ``texture_size(frame)``
    gives the size of a frame's texture."""
    if frame_count == 1:
        width, height = texture_size(0)
        return FrameSize(width, width, height, height)
    if frame_count in (4, 8):
        left, top = texture_size(LT)
        right, bottom = texture_size(RB)
        return FrameSize(left, right, top, bottom)
    return FrameSize(0.0, 0.0, 0.0, 0.0)


def content_quad(base: tuple[float, float], size: Vec2, frame: FrameSize, inflation: WindowInflation) -> Quad:
    """The content area inside the frame, widened by the inflation."""
    return Quad(base[0] + frame.left - inflation.left,
                base[1] - frame.top + inflation.top,
                size.x - frame.left - frame.right + inflation.left + inflation.right,
                size.y - frame.top - frame.bottom + inflation.top + inflation.bottom)


def frame_pieces(frame_count: int, base: tuple[float, float], size: Vec2, frame: FrameSize) -> list[FramePiece]:
    """The pieces the frame is drawn as, in drawing order."""
    x, y = base
    w, h = size.x, size.y
    l, r, t, b = frame
    if frame_count == 1:
        return [FramePiece(0, LT, FLIP_NONE, Quad(x, y, w - r, t)),
                FramePiece(0, RT, FLIP_H, Quad(x + w - r, y, r, h - b)),
                FramePiece(0, RB, FLIP_180, Quad(x + l, y - h + b, w - l, b)),
                FramePiece(0, LB, FLIP_V, Quad(x, y - t, l, h - t))]
    if frame_count == 4:
        return [FramePiece(LT, LT, None, Quad(x, y, w - r, t)),
                FramePiece(RT, RT, None, Quad(x + w - r, y, r, h - b)),
                FramePiece(RB, RB, None, Quad(x + l, y - h + b, w - l, b)),
                FramePiece(LB, LB, None, Quad(x, y - t, l, h - t))]
    if frame_count == 8:
        return [FramePiece(LT, LT, None, Quad(x, y, l, t)),
                FramePiece(T, LT, None, Quad(x + l, y, w - l - r, t)),
                FramePiece(RT, RT, None, Quad(x + w - r, y, r, t)),
                FramePiece(R, RT, None, Quad(x + w - r, y - t, r, h - t - b)),
                FramePiece(RB, RB, None, Quad(x + w - r, y - h + b, r, b)),
                FramePiece(B, RB, None, Quad(x + l, y - h + b, w - l - r, b)),
                FramePiece(LB, LB, None, Quad(x, y - h + b, l, b)),
                FramePiece(L, LB, None, Quad(x, y - t, l, h - t - b))]
    return []


def frame_tex_coords(anchor: int, width: float, height: float, texture_width: int, texture_height: int,
                     flip: int) -> TexCoords:
    """Texture coordinates for a frame piece, so its texture keeps its size from the anchored corner."""
    lt, rt, lb, rb, ix, iy = _FLIPS[min(max(flip, 0), len(_FLIPS) - 1)]
    corners = (lt, rt, lb, rb)
    start, toward_x, toward_y, fixed_x, fixed_y, extended_x, extended_y = _ANCHORS[anchor]

    origin = corners[start]
    uv = [[0.0, 0.0] for _ in range(4)]
    for corner in fixed_x:
        uv[corner][ix] = origin[ix]
    for corner in fixed_y:
        uv[corner][iy] = origin[iy]
    x_end = width / ((corners[toward_x][ix] - origin[ix]) * texture_width) + origin[ix]
    y_end = height / ((corners[toward_y][iy] - origin[iy]) * texture_height) + origin[iy]
    for corner in extended_x:
        uv[corner][ix] = x_end
    for corner in extended_y:
        uv[corner][iy] = y_end
    return TexCoords(*(Vec2(*corner) for corner in uv))
