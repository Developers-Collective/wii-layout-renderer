"""Pane transforms, built the way NW4R's Pane::CalculateMtx builds them.

A pane's matrix is T * Rz * Ry * Rx * S and its global matrix is the parent's global matrix times its own.
Layouts are drawn with an orthographic projection, so a pane's plane (z = 0) maps to the screen through
the X and Y rows of its global 3x4 matrix; rotations about X and Y foreshorten panes like in the game.
"""
from __future__ import annotations

import math
from typing import Optional

from ..state import LayoutState, PaneState, PictureState, TextBoxState, WindowState

# Row-major 3x4 affine matrix: (m00, m01, m02, m03, m10, ..., m23).
Matrix = tuple[float, float, float, float, float, float, float, float, float, float, float, float]
# A rectangle in layout units: left, bottom, right, top (Y up).
Bounds = tuple[float, float, float, float]

IDENTITY: Matrix = (1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0)


def pane_matrix(pane: PaneState) -> Matrix:
    rx, ry, rz = (math.radians(angle) for angle in (pane.rotate.x, pane.rotate.y, pane.rotate.z))
    sin_x, cos_x = math.sin(rx), math.cos(rx)
    sin_y, cos_y = math.sin(ry), math.cos(ry)
    sin_z, cos_z = math.sin(rz), math.cos(rz)
    scale_x, scale_y = pane.scale.x, pane.scale.y
    # Rz * Ry * Rx, like nw4r::math::MTX34RotXYZRad, with the X and Y columns scaled.
    return (cos_z * cos_y * scale_x, (cos_z * sin_y * sin_x - sin_z * cos_x) * scale_y,
            cos_z * sin_y * cos_x + sin_z * sin_x, pane.translate.x,
            sin_z * cos_y * scale_x, (sin_z * sin_y * sin_x + cos_z * cos_x) * scale_y,
            sin_z * sin_y * cos_x - cos_z * sin_x, pane.translate.y,
            -sin_y * scale_x, cos_y * sin_x * scale_y, cos_y * cos_x, pane.translate.z)


def multiply(a: Matrix, b: Matrix) -> Matrix:
    """a * b: transform by b, then by a."""
    result = []
    for row in range(3):
        a0, a1, a2, a3 = a[row * 4:row * 4 + 4]
        result += (a0 * b[0] + a1 * b[4] + a2 * b[8],
                   a0 * b[1] + a1 * b[5] + a2 * b[9],
                   a0 * b[2] + a1 * b[6] + a2 * b[10],
                   a0 * b[3] + a1 * b[7] + a2 * b[11] + a3)
    return tuple(result)


def plane_affine(matrix: Matrix) -> tuple[float, float, float, float, float, float]:
    """How the matrix maps a pane's plane onto the screen: (xx, xy, yx, yy, dx, dy) with
    x' = xx * x + yx * y + dx and y' = xy * x + yy * y + dy, the order QTransform takes."""
    return matrix[0], matrix[4], matrix[1], matrix[5], matrix[3], matrix[7]


def base_point(pane: PaneState) -> tuple[float, float]:
    """The top left corner of the pane's rectangle relative to its origin (Y up)."""
    horizontal, vertical = pane.base_position % 3, pane.base_position // 3
    width, height = pane.size.x, pane.size.y
    x = (0.0, -width / 2, -width)[horizontal]
    y = (0.0, height / 2, height)[min(vertical, 2)]
    return x, y


def drawn_bounds(state: LayoutState) -> Optional[Bounds]:
    """The rectangle around every visible picture, window and text box, or None when nothing is drawn."""
    corners = []

    def visit(pane: PaneState, parent_matrix: Matrix) -> None:
        if not pane.visible:
            return
        matrix = multiply(parent_matrix, pane_matrix(pane))
        if isinstance(pane, (PictureState, WindowState, TextBoxState)):
            xx, xy, yx, yy, dx, dy = plane_affine(matrix)
            x, y = base_point(pane)
            for cx in (x, x + pane.size.x):
                for cy in (y, y - pane.size.y):
                    corners.append((xx * cx + yx * cy + dx, xy * cx + yy * cy + dy))
        for child in pane.children:
            visit(child, matrix)

    if state.root is not None:
        visit(state.root, IDENTITY)
    if not corners:
        return None
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)
