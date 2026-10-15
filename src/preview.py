"""The preview model: which layout is shown, which animations play, the state hooks and the clock.

``LayoutPreview`` does not depend on Qt. ``LayoutPreviewWidget`` draws one on screen, and
``LayoutRenderer`` can render its states without a window.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Union

from .animation import AnimationTrack, apply_animation
from .formats import Layout
from .hooks import FrameContext, HookHandle, StateHook, StateHooks
from .resources import FileSource, FontLibrary, LayoutArchive, Resources
from .state import LayoutState

GAME_FRAME_RATE = 60.0


class Event:
    """Callbacks to call when something happens to a preview; the Qt-free counterpart of a signal."""

    def __init__(self) -> None:
        self._callbacks: list[Callable] = []

    def connect(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback: Callable) -> None:
        self._callbacks.remove(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class LayoutPreview:
    """A layout archive being previewed.

    Animations are tracks: ``set_animation`` shows one BRLAN (looping, playing once, or paused at a frame),
    ``add_animation`` layers more, and later tracks win where they animate the same value. The clock moves
    on with ``advance``; it drives playing tracks and the ``elapsed_frames`` that state hooks see.

    Events (connect callables to them):
      - ``changed()``: the state to draw is different now,
      - ``hook_failed(hook, error)``: a state hook raised and was removed,
      - ``animation_finished(name)``: a track playing once reached its end.
    """

    def __init__(self, *, frame_rate: float = GAME_FRAME_RATE):
        self.changed = Event()
        self.hook_failed = Event()
        self.animation_finished = Event()

        self._fonts = FontLibrary()
        self._resources = Resources(None, self._fonts)
        self._layout_name: Optional[str] = None
        self._layout: Optional[Layout] = None
        self._tracks: list[AnimationTrack] = []
        self._hooks = StateHooks(self._invalidate, self.hook_failed.emit)
        self._frame_rate = frame_rate
        self._elapsed_frames = 0.0
        self._state: Optional[LayoutState] = None

    @classmethod
    def open(cls, source: Union[FileSource, LayoutArchive], layout: Optional[str] = None, *,
             fonts: Optional[Mapping[str, FileSource]] = None, font_dirs: Iterable[FileSource] = ()) -> LayoutPreview:
        preview = cls()
        preview.set_fonts(fonts, font_dirs)
        preview.load_archive(source, layout)
        return preview

    # Content

    @property
    def archive(self) -> Optional[LayoutArchive]:
        return self._resources.archive

    @property
    def resources(self) -> Resources:
        return self._resources

    @property
    def layout_name(self) -> Optional[str]:
        return self._layout_name

    def load_archive(self, source: Union[FileSource, LayoutArchive], layout: Optional[str] = None) -> None:
        """Show a layout archive (a path, its bytes or a ``LayoutArchive``), and its first layout unless
        ``layout`` names another. The animations stop; the state hooks stay."""
        archive = source if isinstance(source, LayoutArchive) else LayoutArchive.open(source)
        if layout is None:
            names = archive.layout_names()
            layout = names[0] if names else None
        model = archive.layout(layout) if layout is not None else None

        self._resources = Resources(archive, self._fonts)
        self._layout_name, self._layout = layout, model
        self._tracks.clear()
        self._invalidate()

    def set_layout(self, name: str) -> None:
        """Show another layout of the archive. The animations keep playing; they bind to panes by name."""
        if self.archive is None:
            raise RuntimeError("No archive is loaded.")
        self._layout = self.archive.layout(name)
        self._layout_name = name
        self._invalidate()

    def set_fonts(self, fonts: Optional[Mapping[str, FileSource]] = None,
                  font_dirs: Iterable[FileSource] = ()) -> None:
        """Fonts for layouts whose archive does not have them, like the game's Font folder. ``fonts`` maps
        font file names to paths or bytes; ``font_dirs`` are searched for the font's file name."""
        self._fonts = FontLibrary(fonts, font_dirs)
        self._resources = Resources(self.archive, self._fonts)
        self._invalidate()

    def layout_names(self) -> list[str]:
        return self.archive.layout_names() if self.archive is not None else []

    def animation_names(self) -> list[str]:
        return self.archive.animation_names() if self.archive is not None else []

    # Animations

    @property
    def tracks(self) -> tuple[AnimationTrack, ...]:
        """The active animations, in the order they are applied."""
        return tuple(self._tracks)

    def track(self, name: str) -> Optional[AnimationTrack]:
        return next((track for track in self._tracks if track.name == name), None)

    def set_animation(self, name: Optional[str], *, frame: float = 0.0, playing: bool = True, loop: bool = True,
                      speed: float = 1.0) -> Optional[AnimationTrack]:
        """Show only this animation: looping (the default), playing once (``loop=False``) or paused at
        ``frame`` (``playing=False``). None shows the layout without animations."""
        self._tracks.clear()
        if name is None:
            self._invalidate()
            return None
        return self.add_animation(name, frame=frame, playing=playing, loop=loop, speed=speed)

    def add_animation(self, name: str, *, frame: float = 0.0, playing: bool = True, loop: bool = True,
                      speed: float = 1.0) -> AnimationTrack:
        """Layer an animation over the active ones, replacing a track of the same name.

        Raises KeyError when the archive has no such file and FormatError when it is damaged."""
        if self.archive is None:
            raise RuntimeError("No archive is loaded.")
        track = AnimationTrack(name, self.archive.animation(name), frame=frame, playing=playing, loop=loop,
                               speed=speed, on_change=self._invalidate)
        self._tracks = [existing for existing in self._tracks if existing.name != name]
        self._tracks.append(track)
        self._invalidate()
        return track

    def remove_animation(self, name: str) -> bool:
        track = self.track(name)
        if track is None:
            return False
        self._tracks.remove(track)
        self._invalidate()
        return True

    def pause_animation(self, frame: Optional[float] = None, name: Optional[str] = None) -> None:
        """Pause the named animation, or all of them, at their current frame or at ``frame``."""
        for track in self._selected_tracks(name):
            track.pause(frame)

    def resume_animation(self, loop: Optional[bool] = None, name: Optional[str] = None) -> None:
        """Play the named animation, or all of them, on from their frame; ``loop`` switches the mode."""
        for track in self._selected_tracks(name):
            track.play(loop=loop)

    def seek_animation(self, frame: float, name: Optional[str] = None) -> None:
        """Move the named animation, or all of them, to ``frame`` without changing whether they play."""
        for track in self._selected_tracks(name):
            track.frame = frame

    def _selected_tracks(self, name: Optional[str]) -> list[AnimationTrack]:
        if name is None:
            return list(self._tracks)
        track = self.track(name)
        if track is None:
            raise KeyError(f"{name!r} is not an active animation.")
        return [track]

    # State hooks

    def add_state_hook(self, hook: StateHook) -> HookHandle:
        """Register ``hook(state, context)`` to change every frame's state before it is drawn (see hooks)."""
        return self._hooks.add(hook)

    def remove_state_hook(self, hook: Union[StateHook, HookHandle]) -> bool:
        return self._hooks.remove(hook)

    def clear_state_hooks(self) -> None:
        self._hooks.clear()

    # Clock

    @property
    def elapsed_frames(self) -> float:
        return self._elapsed_frames

    @property
    def is_animated(self) -> bool:
        """Whether the clock changes what is drawn: an animation plays or a state hook may use the time."""
        return bool(self._hooks) or any(track.playing for track in self._tracks)

    def advance(self, seconds: float) -> None:
        """Move the clock on by real time."""
        self.advance_frames(seconds * self._frame_rate)

    def advance_frames(self, frames: float) -> None:
        """Move the clock, and every playing animation, on by game frames."""
        self._elapsed_frames += frames
        finished = [track.name for track in self._tracks if track.advance(frames)]
        self._invalidate()
        for name in finished:
            self.animation_finished.emit(name)

    def reset_clock(self) -> None:
        self._elapsed_frames = 0.0
        self._invalidate()

    # Output

    def state(self) -> Optional[LayoutState]:
        """The state to draw now: the layout, with the animations applied and then the state hooks run.

        It is built once per frame and shared; None while no layout is loaded."""
        if self._layout is None:
            return None
        if self._state is None:
            state = LayoutState.from_layout(self._layout)
            for track in self._tracks:
                apply_animation(state, track.animation, track.frame)
            context = FrameContext(self._elapsed_frames, self._frame_rate,
                                   tuple(track.info() for track in self._tracks), self._layout_name,
                                   self._resources)
            self._state = self._hooks.run(state, context)
        return self._state

    def _invalidate(self) -> None:
        self._state = None
        self.changed.emit()
