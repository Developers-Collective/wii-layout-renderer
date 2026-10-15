"""A state hook for dialog.arc: lays a text out over the char0..char63 panes like the game's
TextEffectManager does, and lets every character wave.

Run it to see it:  python dialog_text.py path/to/dialog.arc path/to/Font "Hello <wave>world</wave>"

A hook that applies the effects of a speech bubble's .bin file works the same way: lay the characters out,
then move, rotate, scale and colour each character pane from the effect curves at ``context.elapsed_frames``.
"""
from __future__ import annotations

import math
import re
import sys

from wii_layout_renderer import FrameContext, LayoutState, TextBoxState

CHAR_COUNT = 64
TAG = re.compile(r"<(/?)(\w+)[^>]*>")


class DialogText:
    """Shows ``text`` in the dialog's character panes. Characters inside <wave>...</wave> wave."""

    def __init__(self, text: str, amplitude: float = 4.0, speed: float = 0.1, char_offset: float = 0.6):
        self.chars, self.waving = self._parse(text)
        self.amplitude = amplitude
        self.speed = speed
        self.char_offset = char_offset

    @staticmethod
    def _parse(text: str) -> tuple[str, list[bool]]:
        chars, waving, inside = [], [], False
        position = 0
        for match in TAG.finditer(text):
            for char in text[position:match.start()]:
                chars.append(char)
                waving.append(inside)
            inside = match.group(2) == "wave" and not match.group(1)
            position = match.end()
        for char in text[position:]:
            chars.append(char)
            waving.append(inside)
        return "".join(chars), waving

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

        text_box = panes["textbox"]
        left, top = -text_box.size.x / 2, text_box.size.y / 2
        x = y = 0.0
        index = 0
        for char, waving in zip(self.chars, self.waving):
            if char == "\n":
                x, y = 0.0, y - font.line_feed
                continue
            if char == " ":
                x += font.char_widths(" ").char_width
                continue
            if index == CHAR_COUNT:
                break
            widths = font.char_widths(char)
            box = boxes[index]
            box.visible = True
            box.text = char
            box.size.x, box.size.y = widths.char_width, font.line_feed
            box.translate.x, box.translate.y = left + x + widths.left, top + y
            if waving:
                phase = context.elapsed_frames * self.speed + index * self.char_offset
                box.translate.y += self.amplitude * math.sin(phase)
            x += widths.char_width - widths.left
            index += 1


def main() -> None:
    from PyQt6 import QtWidgets

    from wii_layout_renderer import LayoutPreviewWidget

    archive, font_dir, text = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "Hello <wave>world</wave>!"
    app = QtWidgets.QApplication(sys.argv)
    view = LayoutPreviewWidget()
    view.load_archive(archive, font_dirs=[font_dir])
    view.preview.add_state_hook(DialogText(text.replace("\\n", "\n")))
    view.resize(960, 720)
    view.show()
    app.exec()


if __name__ == "__main__":
    main()
