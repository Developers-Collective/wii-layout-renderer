"""State hooks: callbacks that change the render state after the animations and before drawing.

A hook is called as ``hook(state, context)``. It changes the state in place and returns None, or returns a
``LayoutState`` to draw instead. Hooks run in the order they were added.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union

from .animation import TrackInfo
from .resources import Resources
from .state import LayoutState

log = logging.getLogger(__name__)

StateHook = Callable[[LayoutState, "FrameContext"], Optional[LayoutState]]


@dataclass(frozen=True)
class FrameContext:
    """What a state hook knows about the frame being drawn."""

    elapsed_frames: float  # game frames on the preview clock since it was last reset
    frame_rate: float  # game frames per second
    tracks: tuple[TrackInfo, ...]  # the active animations, in the order they are applied
    layout_name: Optional[str]
    resources: Resources  # textures and fonts, e.g. ``resources.font(pane.font).char_widths("A")``

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_frames / self.frame_rate

    def track(self, name: str) -> Optional[TrackInfo]:
        return next((track for track in self.tracks if track.name == name), None)


class HookHandle:
    """Returned by ``add``; ``remove()`` unregisters the hook."""

    def __init__(self, hooks: StateHooks, hook: StateHook):
        self._hooks = hooks
        self.hook = hook

    def remove(self) -> None:
        self._hooks.remove(self)


class StateHooks:
    """The registered state hooks, run in order on every frame's state.

    A hook that raises, or returns something other than None or a ``LayoutState``, is logged and removed so
    the preview keeps drawing; ``on_error(hook, error)`` is told about it.
    """

    def __init__(self, on_change: Callable[[], None], on_error: Callable[[StateHook, Exception], None]):
        self._handles: list[HookHandle] = []
        self._on_change = on_change
        self._on_error = on_error

    def __len__(self) -> int:
        return len(self._handles)

    def add(self, hook: StateHook) -> HookHandle:
        if not callable(hook):
            raise TypeError(f"A state hook must be callable, not {type(hook).__name__}.")
        handle = HookHandle(self, hook)
        self._handles.append(handle)
        self._on_change()
        return handle

    def remove(self, hook: Union[StateHook, HookHandle]) -> bool:
        """Unregister a hook by its handle or by the callable. Returns whether it was registered."""
        for handle in self._handles:
            if handle is hook or handle.hook is hook:
                self._handles.remove(handle)
                self._on_change()
                return True
        return False

    def clear(self) -> None:
        if self._handles:
            self._handles.clear()
            self._on_change()

    def run(self, state: LayoutState, context: FrameContext) -> LayoutState:
        for handle in list(self._handles):
            try:
                result = handle.hook(state, context)
                if result is not None and not isinstance(result, LayoutState):
                    raise TypeError(f"returned {type(result).__name__}, not a LayoutState or None")
            except Exception as error:
                log.exception("State hook %r failed and was removed", handle.hook)
                if handle in self._handles:
                    self._handles.remove(handle)
                self._on_error(handle.hook, error)
                continue
            if result is not None:
                state = result
        return state
