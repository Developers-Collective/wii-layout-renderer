"""Shows a speech bubble the way the game's dDialog_c does: its text laid out over the char0..char63 panes of
dialog.arc, typed in one character every three frames, with the effects its tags name read from TextEffects files.
A picture character (0x0B, 0x0100 with the glyph, 0xBEEF) takes one pane and draws the picture font's glyph.

    python dialog_text.py dialog.arc Font TextEffects "Hello <wave>world</wave>"
    python dialog_text.py dialog.arc Font TextEffects Message/EngEU/SpeechBubbles/4.bin
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from wii_layout_renderer import Color, FrameContext, LayoutState, TextBoxState
from wii_layout_renderer.formats import Brfnt

CHAR_COUNT = 64  # dDialog_c::maxCharCount
PICTURE_FONT = "mj2d00_PictureFont_32_RGBA8.brfnt"
TICKS_PER_CHAR = 3  # dDialog_c::tickCharOffset
WHITESPACE = " \n\r\t"
CLAMP, REPEAT, PING_PONG = 0, 1, 2

Rgba = tuple[int, ...]
WHITE: Rgba = (255, 255, 255, 255)


# Effect files: TextEffects/<name>.bin, big endian and packed.

class Reader:
    def __init__(self, data: bytes):
        self.data, self.position = data, 0

    def values(self, fmt: str) -> tuple:
        values = struct.unpack_from(">" + fmt, self.data, self.position)
        self.position += struct.calcsize(">" + fmt)
        return values

    def value(self, fmt: str):
        return self.values(fmt)[0]

    def cstring(self) -> str:
        end = self.data.index(b"\0", self.position)
        text = self.data[self.position:end].decode("ascii")
        self.position = end + 1
        return text


class KeyGroup:
    """Keys sorted by time; ``key[0]`` is the time and ``key[1]`` the value."""

    def __init__(self, loop: int, keys: list):
        self.loop, self.keys = loop, keys

    @property
    def duration(self) -> float:
        return self.keys[-1][0] if self.keys else 0.0

    def fixed_time(self, time: float, loop: Optional[int] = None) -> float:
        loop = self.loop if loop is None else loop
        duration = self.duration
        if loop in (REPEAT, PING_PONG) and duration == 0.0:
            return 0.0
        if loop == REPEAT:
            time = math.fmod(time, duration)
            return time + duration if time < 0 else time
        if loop == PING_PONG:
            odd = math.fmod(int(time / duration), 2) == 1
            time = self.fixed_time(time, REPEAT)
            return duration - time if odd else time
        return min(max(time, 0.0), duration)

    def segment(self, time: float):
        """The key value when the time is outside the keys, else the two keys around it."""
        time = self.fixed_time(time)
        if time <= self.keys[0][0]:
            return self.keys[0][1], None, time
        if time >= self.keys[-1][0]:
            return self.keys[-1][1], None, time
        for left, right in zip(self.keys, self.keys[1:]):
            if left[0] <= time <= right[0]:
                return left, right, time
        return self.keys[-1][1], None, time


class Curve(KeyGroup):
    """Keys of time, value, in tangent, out tangent (and weights, which the game ignores)."""

    @classmethod
    def read(cls, reader: Reader) -> Curve:
        loop, count = reader.values("BI")
        return cls(loop, [reader.values("6f") for _ in range(count)])

    def evaluate(self, time: float) -> float:
        if not self.keys:
            return 0.0
        left, right, time = self.segment(time)
        if right is None:
            return left
        span = right[0] - left[0]
        t = (time - left[0]) / span
        return ((2 * t ** 3 - 3 * t ** 2 + 1) * left[1] + (-2 * t ** 3 + 3 * t ** 2) * right[1]
                + (t ** 3 - 2 * t ** 2 + t) * left[3] * span + (t ** 3 - t ** 2) * right[2] * span)


class Gradient(KeyGroup):
    """Keys of time and an RGBA colour."""

    @classmethod
    def read(cls, reader: Reader) -> Gradient:
        loop, count = reader.values("BI")
        keys = []
        for _ in range(count):
            time, *rgba = reader.values("f4B")
            keys.append((time, tuple(rgba)))
        return cls(loop, keys)

    def evaluate(self, time: float) -> Rgba:
        if not self.keys:
            return WHITE
        left, right, time = self.segment(time)
        if right is None:
            return left
        fraction = (time - left[0]) / (right[0] - left[0])
        return tuple(int(a + (b - a) * fraction) for a, b in zip(left[1], right[1]))


@dataclass
class Wave:
    """How one value of a character moves: a curve run at a speed, offset in time per character."""

    curve: Curve
    speed: float
    amplitude: float
    char_time_offset: float
    reverse: bool

    @classmethod
    def read(cls, reader: Reader) -> Wave:
        curve = Curve.read(reader)
        speed, amplitude, char_time_offset, reverse = reader.values("ffd?")
        return cls(curve, speed, amplitude, char_time_offset, reverse)

    def value(self, time: float, index: int) -> float:
        offset = index * self.char_time_offset * (-1.0 if self.reverse else 1.0)
        return self.curve.evaluate(time * self.speed + offset) * self.amplitude


@dataclass
class ColorWave:
    top: Optional[Gradient]
    bottom: Optional[Gradient]
    speed: float
    char_time_offset: float
    reverse: bool


@dataclass
class CharMesh:
    """A character's pane values on top of where the text puts it, reset every frame."""

    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    top: Rgba = WHITE
    bottom: Rgba = WHITE


def _optional(reader: Reader, read):
    return read(reader) if reader.value("?") else None


@dataclass
class TextEffect:
    name: str
    attack: Curve
    continue_forever: bool
    decay: Optional[Curve]
    movement: tuple[Optional[Wave], Optional[Wave]] = (None, None)
    rotation: Optional[Wave] = None
    scale: tuple[Optional[Wave], Optional[Wave]] = (None, None)
    color: Optional[ColorWave] = None
    duration: float = field(init=False, default=0.0)

    @classmethod
    def read(cls, data: bytes) -> TextEffect:
        reader = Reader(data)
        name = reader.cstring()
        reader.value("?")  # time spawn dependant, unused
        attack = Curve.read(reader)
        continue_forever = reader.value("?")
        decay, cycles = (None, 0) if continue_forever else (Curve.read(reader), reader.value("I"))
        movement = _optional(reader, lambda r: (_optional(r, Wave.read), _optional(r, Wave.read))) or (None, None)
        rotation = _optional(reader, lambda r: _optional(r, Wave.read))
        scale = _optional(reader, lambda r: (_optional(r, Wave.read), _optional(r, Wave.read))) or (None, None)
        color = _optional(reader, lambda r: ColorWave(_optional(r, Gradient.read), _optional(r, Gradient.read),
                                                      *r.values("fd?")))
        effect = cls(name, attack, continue_forever, decay, movement, rotation, scale, color)
        effect._place_decay(cycles)
        return effect

    def _place_decay(self, cycles: int) -> None:
        waves = [*self.movement, self.rotation, *self.scale]
        curves = [self.attack, self.decay, *(wave.curve for wave in waves if wave is not None)]
        if self.color is not None:
            curves += [self.color.top, self.color.bottom]
        self.duration = max(curve.duration for curve in curves if curve is not None)
        decay = self.decay
        if decay is None:
            return
        self.duration = self.duration * cycles - decay.duration
        decay.keys = [(key[0] + self.duration, *key[1:]) for key in decay.keys]

    def apply(self, mesh: CharMesh, index: int, time: float, multiplier: float) -> None:
        strength = self.attack.evaluate(time)
        if self.decay is not None:  # the effect ends
            strength *= self.decay.evaluate(time)
        strength *= multiplier

        x, y = self.movement
        if x is not None:
            mesh.x += x.value(time, index) * strength
        if y is not None:
            mesh.y += y.value(time, index) * strength
        if self.rotation is not None:
            mesh.rotation += self.rotation.value(time, index) * strength
        x, y = self.scale
        if x is not None:
            mesh.scale_x *= x.value(time, index) * strength
        if y is not None:
            mesh.scale_y *= y.value(time, index) * strength
        if self.color is not None:
            _apply_color(self.color, mesh, index, time, min(max(strength, 0.0), 1.0))


def _apply_color(color: ColorWave, mesh: CharMesh, index: int, time: float, strength: float) -> None:
    time = time * color.speed + index * color.char_time_offset * (-1.0 if color.reverse else 1.0)

    def blend(current: Rgba, gradient: Gradient) -> Rgba:
        return tuple(int(a * (1.0 - strength) + b * strength) for a, b in zip(current, gradient.evaluate(time)))

    if color.top is not None:
        mesh.top = blend(mesh.top, color.top)
    if color.bottom is not None:
        mesh.bottom = blend(mesh.bottom, color.bottom)


# Tags and the text layout, as TextEffectManager parses them.

@dataclass
class Tag:
    name: str
    mode: int  # 0 opening, 1 closing, 2 self-closing
    length: int  # up to the >


def parse_tag(text: str, position: int) -> Optional[Tag]:
    def at(index):
        return text[index] if index < len(text) else "\0"

    if at(position) != "<":
        return None
    index, mode = position + 1, 0
    if at(index) == "/":
        mode, index = 1, index + 1
    start = index
    while at(index).isascii() and at(index).isalpha():
        index += 1
    name = text[start:index]
    while at(index).isspace():
        index += 1
    if at(index) == "=":
        index += 1
        while at(index).isspace():
            index += 1
        quoted = at(index) == '"'
        index += quoted
        while (quoted and at(index) not in '"\0') or at(index).isdigit() or at(index) == ".":
            index += 1
        index += quoted and at(index) == '"'
    while at(index).isspace():
        index += 1
    if mode == 0 and at(index) == "/":
        mode, index = 2, index + 1
    end = text.find(">", index)
    return Tag(name, mode, end - position) if end >= 0 else None


def load_effects(text: str, effects_dir: Path) -> dict[str, TextEffect]:
    """TextEffects/<tag>.bin once for every tag of the text whose file exists."""
    effects = {}
    for position, char in enumerate(text):
        tag = parse_tag(text, position) if char == "<" else None
        if tag is None or not tag.name or tag.name in effects:
            continue
        path = effects_dir / f"{tag.name}.bin"
        if path.is_file():
            effect = TextEffect.read(path.read_bytes())
            effects[effect.name] = effect
    return effects


def effect_tag(text: str, position: int, effects: dict[str, TextEffect]) -> Optional[Tag]:
    """The tag of a loaded effect at the position; any other tag is drawn as text."""
    tag = parse_tag(text, position) if text[position] == "<" else None
    return tag if tag is not None and tag.name in effects else None


def char_length(text: str, position: int) -> int:
    """How many units the character at position takes: a picture character's three, else one."""
    picture = (text[position:position + 1] == "\x0b" and 0x100 <= ord(text[position + 1:position + 2] or "\0") <= 0x1FF
               and text[position + 2:position + 3] == "\ubeef")
    return 3 if picture else 1


def drawn_char_count(text: str, effects: dict[str, TextEffect]) -> int:
    count, position = 0, 0
    while position < len(text):
        tag = effect_tag(text, position, effects)
        if tag is not None:
            position += tag.length + 1
            continue
        count += text[position] not in WHITESPACE
        position += char_length(text, position)
    return count


@dataclass
class LaidOutChar:
    char: str  # one character, or a picture character's three units
    x: float
    y: float
    width: float

    @property
    def picture(self) -> Optional[int]:
        """The picture font glyph a picture character draws, or None."""
        return ord(self.char[1]) & 0xFF if len(self.char) == 3 else None


@dataclass
class TextLayout:
    chars: list[LaidOutChar]
    ranges: list[tuple[TextEffect, int, int]]  # an effect and the characters it moves, in the order they apply


def lay_out(text: str, effects: dict[str, TextEffect], font: Brfnt, picture_font: Optional[Brfnt],
            box_width: float, box_height: float) -> Optional[TextLayout]:
    """Where dDialog_c::setText and TextEffectManager::parseText put each character. None when the text does not
    fit the layout's character panes, which makes the game show nothing. Picture characters advance by their
    picture font glyph, or like their first unit without the picture font."""
    if drawn_char_count(text, effects) > CHAR_COUNT:
        return None
    left, top = -box_width / 2, box_height / 2
    # Pane k starts with the text's k-th character and pane 0 with a space; before a pane takes its character,
    # its old character's width is the width a following space advances.
    pane_chars = [" "] + list(text[1:CHAR_COUNT])
    space_width = font.char_widths(" ").char_width
    x = y = 0.0
    chars: list[LaidOutChar] = []
    ranges: list[tuple[TextEffect, int, int]] = []
    opened: list[tuple[TextEffect, int]] = []
    position = 0
    while position < len(text):
        tag = effect_tag(text, position, effects)
        if tag is not None:
            effect = effects[tag.name]
            if tag.mode == 0:
                if any(open_effect is effect for open_effect, _ in opened):
                    break
                opened.append((effect, len(chars)))
            elif tag.mode == 1:
                if not opened or opened[-1][0] is not effect:
                    break
                ranges.append((effect, opened.pop()[1], len(chars)))
            position += tag.length + 1
            continue

        char = text[position:position + char_length(text, position)]
        position += len(char)
        if char in "\r\t":
            continue
        if char == "\n":
            x, y = 0.0, y - font.line_feed
            continue
        if char == " ":
            x += space_width
            continue
        index = len(chars)
        space_width = font.char_widths(pane_chars[index] if index < len(pane_chars) else "\0").char_width
        if len(char) == 3 and picture_font is not None:
            widths = picture_font.char_widths(ord(char[1]) & 0xFF)
        else:
            widths = font.char_widths(char[0])
        chars.append(LaidOutChar(char, left + x + widths.left, top + y, widths.char_width))
        x += widths.char_width - widths.left
    return TextLayout(chars, ranges)


# The state hook

class SpeechBubble:
    """Shows ``text`` in dialog.arc. With ``typing``, one character appears every three frames like in the game."""

    def __init__(self, text: str, effects_dir: Path, typing: bool = True):
        self.text = text
        self.effects = load_effects(text, Path(effects_dir))
        self.typing = typing
        self._layout: Optional[TextLayout] = None
        self._laid_out = False

    def __call__(self, state: LayoutState, context: FrameContext) -> None:
        panes = state.panes_by_name()
        boxes = [panes.get(f"char{index}") for index in range(CHAR_COUNT)]
        if not all(isinstance(box, TextBoxState) for box in boxes):
            return
        for box in boxes:
            box.visible = False
        font = context.resources.font(boxes[0].font)
        if font is None:
            return
        if not self._laid_out:
            text_box = panes["textbox"]
            self._layout = lay_out(self.text, self.effects, font, context.resources.font(PICTURE_FONT),
                                   text_box.size.x, text_box.size.y)
            self._laid_out = True
        if self._layout is None:
            return

        frame = context.elapsed_frames
        meshes = [CharMesh() for _ in self._layout.chars]
        for effect, start, end in self._layout.ranges:
            for index in range(start, end):
                effect.apply(meshes[index], index, frame / 60.0, 1.0)

        for index, (char, mesh) in enumerate(zip(self._layout.chars, meshes)):
            box = boxes[index]
            box.visible = not self.typing or index * TICKS_PER_CHAR <= frame
            if char.picture is not None:
                box.font, box.text = PICTURE_FONT, chr(char.picture)
            else:
                box.text = char.char
            box.size.x, box.size.y = char.width, font.line_feed
            box.translate.x, box.translate.y = char.x + mesh.x, char.y + mesh.y
            box.translate.z = mesh.rotation  # the game keeps the rotation there, where it does not show
            box.scale.x, box.scale.y = mesh.scale_x, mesh.scale_y
            box.top_color, box.bottom_color = Color(*mesh.top), Color(*mesh.bottom)


def read_message(path: Path) -> str:
    """The text of a Message/<language>/SpeechBubbles/<ID>.bin file: u16 style, u16 0, u32 text offset."""
    data = path.read_bytes()
    offset = struct.unpack_from(">I", data, 4)[0]
    end = offset
    while end + 1 < len(data) and data[end:end + 2] != b"\0\0":
        end += 2
    return data[offset:end].decode("utf-16-be")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shows a speech bubble of dialog.arc with its text effects.")
    parser.add_argument("archive", help="the speech bubble layout, e.g. Layout/dialog.arc")
    parser.add_argument("font_dir", help="the folder with the layout's font, e.g. the game's Font folder")
    parser.add_argument("effects_dir", help="the TextEffects folder")
    parser.add_argument("text", help="the text, or a SpeechBubbles/<ID>.bin file")
    parser.add_argument("--no-typing", action="store_true", help="show the whole text at once")
    args = parser.parse_args()

    from PyQt6 import QtWidgets
    from wii_layout_renderer import LayoutPreviewWidget

    text = read_message(Path(args.text)) if args.text.endswith(".bin") else args.text.replace("\\n", "\n")
    app = QtWidgets.QApplication(sys.argv)
    view = LayoutPreviewWidget()
    view.load_archive(args.archive, font_dirs=[args.font_dir])
    view.preview.add_state_hook(SpeechBubble(text, Path(args.effects_dir), typing=not args.no_typing))
    view.resize(960, 720)
    view.show()
    app.exec()


if __name__ == "__main__":
    main()
