"""Applying a BRLAN at a frame to a render state. Contents bind to panes and materials by name."""
from __future__ import annotations

from typing import Callable

from ..formats.brlan import MATERIAL_CONTENT, PANE_CONTENT, AnimationTarget, LayoutAnimation
from ..state import (Color, LayoutState, MaterialState, PaneState, PictureState, TextBoxState, TextureSrt,
                     WindowState)
from .curves import evaluate

# Value setters by animation kind; each takes the pane or material, the target and the evaluated value.
Setter = Callable[[object, AnimationTarget, float, LayoutAnimation], None]


def apply_animation(state: LayoutState, animation: LayoutAnimation, frame: float) -> None:
    """Set every animated value of ``state`` to its value at ``frame``."""
    panes = state.panes_by_name()
    materials: dict[str, MaterialState] = {}
    for material in state.materials:
        materials.setdefault(material.name, material)

    for content in animation.contents:
        if content.type == PANE_CONTENT:
            subject, setters = panes.get(content.name), _PANE_SETTERS
        elif content.type == MATERIAL_CONTENT:
            subject, setters = materials.get(content.name), _MATERIAL_SETTERS
        else:
            continue
        if subject is None:
            continue
        for info in content.infos:
            setter = setters.get(info.kind)
            if setter is None:
                continue
            for target in info.targets:
                if target.keys:
                    setter(subject, target, evaluate(target, frame), animation)


def _to_byte(value: float) -> int:
    return min(255, max(0, round(value)))


def _to_s10(value: float) -> int:
    return min(1023, max(-1024, round(value)))


def _set_channel(color: Color, channel: int, value: int) -> None:
    setattr(color, "rgba"[channel], value)


# Pane SRT (RLPA): translate, rotate, scale, size.
_SRT_FIELDS = (("translate", "x"), ("translate", "y"), ("translate", "z"),
               ("rotate", "x"), ("rotate", "y"), ("rotate", "z"),
               ("scale", "x"), ("scale", "y"), ("size", "x"), ("size", "y"))


def _set_pane_srt(pane: PaneState, target: AnimationTarget, value: float, _) -> None:
    if target.target < len(_SRT_FIELDS):
        vector, axis = _SRT_FIELDS[target.target]
        setattr(getattr(pane, vector), axis, value)


def _set_pane_visibility(pane: PaneState, target: AnimationTarget, value: float, _) -> None:
    if target.target == 0:
        pane.visible = value != 0.0


def _set_pane_color(pane: PaneState, target: AnimationTarget, value: float, _) -> None:
    """RLVC: targets 0-15 are the four corner colours' RGBA channels, 16 is the pane alpha. A text box
    takes corners 0 and 1 as its top colour and 2 and 3 as its bottom colour."""
    if target.target >= 16:
        pane.alpha = _to_byte(value)
        return
    corner, channel = divmod(target.target, 4)
    if isinstance(pane, PictureState):
        color = pane.vertex_colors[corner]
    elif isinstance(pane, WindowState):
        color = pane.content_vertex_colors[corner]
    elif isinstance(pane, TextBoxState):
        color = pane.top_color if corner < 2 else pane.bottom_color
    else:
        return
    _set_channel(color, channel, _to_byte(value))


_PANE_SETTERS: dict[str, Setter] = {
    "RLPA": _set_pane_srt,
    "RLVI": _set_pane_visibility,
    "RLVC": _set_pane_color,
}


def _set_texture_pattern(material: MaterialState, target: AnimationTarget, value: float,
                         animation: LayoutAnimation) -> None:
    """RLTP: the value picks a texture from the animation's file list for one texture map."""
    file_index = max(0, round(value))
    if target.target == 0 and target.index < len(material.texture_maps) and file_index < len(animation.files):
        material.texture_maps[target.index].texture = animation.files[file_index]


_TEXTURE_SRT_FIELDS = (("translate", "x"), ("translate", "y"), ("rotate", None), ("scale", "x"), ("scale", "y"))


def _set_texture_srt(material: MaterialState, target: AnimationTarget, value: float, _) -> None:
    if target.index >= len(material.texture_srts) or target.target >= len(_TEXTURE_SRT_FIELDS):
        return
    srt: TextureSrt = material.texture_srts[target.index]
    name, axis = _TEXTURE_SRT_FIELDS[target.target]
    if axis is None:
        srt.rotate = value
    else:
        setattr(getattr(srt, name), axis, value)


def _set_material_color(material: MaterialState, target: AnimationTarget, value: float, _) -> None:
    """RLMC: groups 1-3 of four channels are the TEV colour registers (0 is the material colour and 4-7 the
    constant colours, which are not drawn)."""
    group, channel = divmod(target.target, 4)
    if 1 <= group <= 3:
        _set_channel(material.tev_colors[group - 1], channel, _to_s10(value))


_MATERIAL_SETTERS: dict[str, Setter] = {
    "RLTP": _set_texture_pattern,
    "RLTS": _set_texture_srt,
    "RLMC": _set_material_color,
}
