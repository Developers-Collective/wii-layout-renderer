"""Evaluating BRLAN keyframe curves."""
from __future__ import annotations

from ..formats.brlan import AnimationTarget


def evaluate(target: AnimationTarget, frame: float) -> float:
    """A target's value at a frame.

    Values hold before the first and after the last key. Step curves hold each key until the next one;
    Hermite curves blend cubically, the slopes being per-frame tangents scaled by the key interval.
    """
    keys = target.keys
    if not keys:
        return 0.0
    if frame <= keys[0].frame:
        return keys[0].value
    if frame >= keys[-1].frame:
        return keys[-1].value

    for a, b in zip(keys, keys[1:]):
        if not a.frame <= frame <= b.frame:
            continue
        if target.is_step:
            return a.value
        span = b.frame - a.frame
        if span <= 0.0:
            return a.value
        t = (frame - a.frame) / span
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return h00 * a.value + h10 * span * a.slope + h01 * b.value + h11 * span * b.slope

    return keys[-1].value
