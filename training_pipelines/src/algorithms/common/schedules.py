
"""Parameter schedules (e.g. epsilon-greedy exploration decay)."""
from __future__ import annotations

from typing import Callable


def linear_schedule(start: float, end: float, decay_steps: int) -> Callable[[int], float]:
    """Return step -> value, linearly interpolated from `start` to `end`."""
    decay_steps = max(1, int(decay_steps))

    def _schedule(step: int) -> float:
        frac = min(1.0, max(0.0, step / decay_steps))
        return start + frac * (end - start)

    return _schedule