"""The preview widget: shows a ``LayoutPreview`` and runs its clock while something animates."""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Union

from PyQt6 import QtCore, QtGui, QtWidgets

from .preview import LayoutPreview
from .render import LayoutRenderer
from .render.geometry import Bounds
from .render.renderer import DEFAULT_CANVAS_SIZE, canvas_size
from .resources import FileSource, LayoutArchive

MAX_CLOCK_STEP = 0.25  # seconds; longer stalls (a breakpoint, a dragged window) do not skip the animation ahead


class LayoutPreviewWidget(QtWidgets.QWidget):
    """Draws a layout the way the game would, read-only.

    What is shown is controlled through ``preview`` (a ``LayoutPreview``): the archive and layout, the
    animations (``set_animation``, ``add_animation``, ``pause_animation``, ...) and the state hooks
    (``add_state_hook``). The widget redraws whenever that changes, and while an animation plays or a
    state hook is registered it moves the preview's clock on with real time. Its height follows its width, keeping
    the proportions of what it shows.
    """

    hook_failed = QtCore.pyqtSignal(object, object)  # hook, exception: the hook raised and was removed
    animation_finished = QtCore.pyqtSignal(str)  # animation name: a track playing once reached its end

    def __init__(self, preview: Optional[LayoutPreview] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._renderer = LayoutRenderer()
        self._preview: Optional[LayoutPreview] = None
        self._background: Optional[QtGui.QColor] = QtGui.QColor(0, 0, 0)
        self._display_aspect: Optional[float] = None
        self._view_area: Optional[Bounds] = None
        self._running = True
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._clock = QtCore.QElapsedTimer()

        self.set_preview(preview if preview is not None else LayoutPreview())

    # What is shown

    @property
    def preview(self) -> LayoutPreview:
        return self._preview

    def set_preview(self, preview: LayoutPreview) -> None:
        if self._preview is not None:
            self._preview.changed.disconnect(self._preview_changed)
            self._preview.hook_failed.disconnect(self.hook_failed.emit)
            self._preview.animation_finished.disconnect(self.animation_finished.emit)
        self._preview = preview
        preview.changed.connect(self._preview_changed)
        preview.hook_failed.connect(self.hook_failed.emit)
        preview.animation_finished.connect(self.animation_finished.emit)
        self._preview_changed()
        self.updateGeometry()

    def load_archive(self, source: Union[FileSource, LayoutArchive], layout: Optional[str] = None, *,
                     fonts: Optional[Mapping[str, FileSource]] = None,
                     font_dirs: Optional[Iterable[FileSource]] = None) -> None:
        """Show a layout archive. ``fonts`` and ``font_dirs``, when given, replace the fonts available to
        layouts whose archive does not hold their font (see ``LayoutPreview.set_fonts``)."""
        if fonts is not None or font_dirs is not None:
            self._preview.set_fonts(fonts, font_dirs or ())
        self._preview.load_archive(source, layout)
        self.updateGeometry()

    # Display

    def set_background(self, color: Optional[QtGui.QColor]) -> None:
        """The colour behind the layout's canvas, or None to leave it transparent."""
        self._background = QtGui.QColor(color) if color is not None else None
        self.update()

    def set_display_aspect(self, aspect: Optional[float]) -> None:
        """Width / height to show the canvas at, e.g. 16 / 9 to stretch it like a widescreen TV does;
        None keeps the canvas' own proportions."""
        self._display_aspect = aspect
        self.updateGeometry()
        self.update()

    def set_view_area(self, area: Optional[Bounds]) -> None:
        """Show only part of the canvas: (left, bottom, right, top) in layout units, e.g. around what
        ``drawn_bounds`` measures. None shows the whole canvas."""
        if area is not None and (area[2] <= area[0] or area[3] <= area[1]):
            raise ValueError(f"The view area {area} is empty.")
        self._view_area = area
        self.updateGeometry()
        self.update()

    def shown_rect(self) -> QtCore.QRectF:
        """Where the shown area is drawn: as large as fits, centered."""
        width, height = self._shown_size()
        scale = min(self.width() / width, self.height() / height)
        rect = QtCore.QRectF(0.0, 0.0, width * scale, height * scale)
        rect.moveCenter(QtCore.QRectF(self.rect()).center())
        return rect

    def canvas_rect(self) -> QtCore.QRectF:
        """Where the whole canvas is drawn. With a view area it reaches past what is shown."""
        left, bottom, right, top = self._shown_area()
        canvas_width, canvas_height = self._canvas_size()
        shown = self.shown_rect()
        scale_x = shown.width() / (right - left)
        scale_y = shown.height() / (top - bottom)
        return QtCore.QRectF(shown.left() + (-canvas_width / 2 - left) * scale_x,
                             shown.top() + (top - canvas_height / 2) * scale_y,
                             canvas_width * scale_x, canvas_height * scale_y)

    def _canvas_size(self) -> tuple[float, float]:
        state = self._preview.state() if self._preview is not None else None
        return canvas_size(state) if state is not None else DEFAULT_CANVAS_SIZE

    def _shown_area(self) -> Bounds:
        if self._view_area is not None:
            return self._view_area
        width, height = self._canvas_size()
        return -width / 2, -height / 2, width / 2, height / 2

    def _shown_size(self) -> tuple[float, float]:
        """The shown area's size in layout units, its width stretched by the display aspect."""
        left, bottom, right, top = self._shown_area()
        canvas_width, canvas_height = self._canvas_size()
        stretch = self._display_aspect / (canvas_width / canvas_height) if self._display_aspect else 1.0
        return (right - left) * stretch, top - bottom

    def render_image(self, width: Optional[int] = None, height: Optional[int] = None) -> QtGui.QImage:
        """The current frame as an image, by default at the canvas size."""
        state = self._preview.state()
        if state is None:
            return QtGui.QImage()
        canvas_width, canvas_height = canvas_size(state)
        return self._renderer.render_image(state, width or round(canvas_width), height or round(canvas_height),
                                           self._background)

    # Clock

    def set_running(self, running: bool) -> None:
        """Stop or restart the clock. Stopped, nothing moves: animations and state hooks see a frozen time."""
        self._running = running
        self._update_timer()

    def _tick(self) -> None:
        seconds = self._clock.restart() / 1000.0
        self._preview.advance(min(seconds, MAX_CLOCK_STEP))

    def _update_timer(self) -> None:
        should_run = self._running and self.isVisible() and self._preview.is_animated
        if should_run and not self._timer.isActive():
            self._clock.start()
            self._timer.start()
        elif not should_run and self._timer.isActive():
            self._timer.stop()

    def _preview_changed(self) -> None:
        self._renderer.resources = self._preview.resources
        self._update_timer()
        self.update()

    # Qt

    def sizeHint(self) -> QtCore.QSize:
        width, height = self._shown_size()
        return QtCore.QSize(round(width), round(height))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        shown_width, shown_height = self._shown_size()
        return round(width * shown_height / shown_width)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        state = self._preview.state()
        if state is None:
            return
        painter = QtGui.QPainter(self)
        try:
            painter.setClipRect(self.shown_rect())
            self._renderer.paint(painter, state, self.canvas_rect(), self._background)
        finally:
            painter.end()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._update_timer()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self._update_timer()
