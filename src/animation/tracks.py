"""Animation tracks: a BRLAN bound to a preview, with its own frame and play mode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..formats.brlan import LayoutAnimation


@dataclass(frozen=True)
class TrackInfo:
    """A snapshot of a track, as state hooks see it."""

    name: str
    frame: float
    length: float
    playing: bool
    loop: bool


class AnimationTrack:
    """One BRLAN playing on the layout.

    A track is paused or playing; a playing track either loops or plays once and then pauses on its last
    frame. Frames run from 0 to ``length`` (the BRLAN's frame size); looping wraps ``length`` back to 0.
    Changing a track redraws the preview it belongs to.
    """

    def __init__(self, name: str, animation: LayoutAnimation, *, frame: float = 0.0, playing: bool = True,
                 loop: bool = True, speed: float = 1.0, on_change: Optional[Callable[[], None]] = None):
        self._name = name
        self._animation = animation
        self._playing = playing
        self._loop = loop
        self._speed = speed
        self._on_change = on_change
        self._frame = self._fit(frame)

    def __repr__(self) -> str:
        mode = ("looping" if self._loop else "playing once") if self._playing else "paused"
        return f"<AnimationTrack {self._name!r} {mode} at frame {self._frame:g}/{self.length:g}>"

    @property
    def name(self) -> str:
        return self._name

    @property
    def animation(self) -> LayoutAnimation:
        return self._animation

    @property
    def length(self) -> float:
        return float(self._animation.frame_size)

    @property
    def frame(self) -> float:
        return self._frame

    @frame.setter
    def frame(self, frame: float) -> None:
        self._frame = self._fit(frame)
        self._changed()

    @property
    def playing(self) -> bool:
        return self._playing

    @playing.setter
    def playing(self, playing: bool) -> None:
        self._playing = playing
        self._changed()

    @property
    def loop(self) -> bool:
        return self._loop

    @loop.setter
    def loop(self, loop: bool) -> None:
        self._loop = loop
        self._frame = self._fit(self._frame)
        self._changed()

    @property
    def speed(self) -> float:
        """Animation frames per game frame (1 plays at the game's speed)."""
        return self._speed

    @speed.setter
    def speed(self, speed: float) -> None:
        self._speed = speed
        self._changed()

    def pause(self, frame: Optional[float] = None) -> None:
        """Stop at the current frame, or at ``frame``."""
        self._playing = False
        if frame is not None:
            self._frame = self._fit(frame)
        self._changed()

    def play(self, *, loop: Optional[bool] = None, frame: Optional[float] = None) -> None:
        """Play from the current frame, or from ``frame``; ``loop`` switches between looping and playing once."""
        if loop is not None:
            self._loop = loop
        if frame is not None:
            self._frame = self._fit(frame)
        elif not self._loop and self._frame >= self.length:
            self._frame = 0.0  # replaying a finished one-shot starts it over
        self._playing = True
        self._changed()

    def info(self) -> TrackInfo:
        return TrackInfo(self._name, self._frame, self.length, self._playing, self._loop)

    def advance(self, frames: float) -> bool:
        """Move a playing track ``frames`` game frames on. Returns True when a one-shot track just ended."""
        step = frames * self._speed
        if not self._playing or step == 0.0:
            return False
        frame = self._frame + step
        finished = not self._loop and (frame >= self.length if step > 0.0 else frame <= 0.0)
        self._frame = self._fit(frame)
        if finished:
            self._playing = False
        return finished

    def _fit(self, frame: float) -> float:
        length = self.length
        if length <= 0.0:
            return 0.0
        if self._loop:
            return frame % length
        return min(max(frame, 0.0), length)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()
