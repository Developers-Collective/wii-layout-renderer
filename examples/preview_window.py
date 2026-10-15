"""A small viewer for a layout archive, with animation controls.

    python preview_window.py path/to/layout.arc [path/to/Font]
"""
from __future__ import annotations

import sys

from PyQt6 import QtCore, QtWidgets

from wii_layout_renderer import LayoutPreviewWidget

NO_ANIMATION = "(no animation)"


class PreviewWindow(QtWidgets.QWidget):
    def __init__(self, archive: str, font_dirs: list[str]):
        super().__init__()
        self.view = LayoutPreviewWidget()
        self.view.load_archive(archive, font_dirs=font_dirs)
        self.preview = self.view.preview

        self.layouts = QtWidgets.QComboBox()
        self.layouts.addItems(self.preview.layout_names())
        self.layouts.currentTextChanged.connect(self.preview.set_layout)

        self.animations = QtWidgets.QComboBox()
        self.animations.addItems([NO_ANIMATION] + self.preview.animation_names())
        self.animations.currentTextChanged.connect(self.choose_animation)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Loop", "Play once", "Paused"])
        self.mode.currentTextChanged.connect(lambda _: self.choose_animation(self.animations.currentText()))

        self.frame = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame.valueChanged.connect(self.seek)
        self.frame_label = QtWidgets.QLabel()

        controls = QtWidgets.QHBoxLayout()
        for widget in (self.layouts, self.animations, self.mode, self.frame, self.frame_label):
            controls.addWidget(widget)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addLayout(controls)

        self.preview.changed.connect(self.show_frame)
        self.choose_animation(self.animations.currentText())

    def choose_animation(self, name: str) -> None:
        if name == NO_ANIMATION:
            self.preview.set_animation(None)
            return
        mode = self.mode.currentText()
        try:
            track = self.preview.set_animation(name, loop=mode == "Loop", playing=mode != "Paused",
                                               frame=self.frame.value() if mode == "Paused" else 0)
        except (KeyError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Animation", f"{name} cannot be played: {error}")
            return
        self.frame.setRange(0, int(track.length))

    def seek(self, frame: int) -> None:
        if self.preview.tracks:
            self.preview.pause_animation(frame)
            self.mode.blockSignals(True)
            self.mode.setCurrentText("Paused")
            self.mode.blockSignals(False)

    def show_frame(self) -> None:
        tracks = self.preview.tracks
        self.frame_label.setText(f"{tracks[0].frame:6.1f} / {tracks[0].length:g}" if tracks else "")
        if tracks and tracks[0].playing:
            self.frame.blockSignals(True)
            self.frame.setValue(int(tracks[0].frame))
            self.frame.blockSignals(False)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = PreviewWindow(sys.argv[1], sys.argv[2:])
    window.resize(960, 760)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
