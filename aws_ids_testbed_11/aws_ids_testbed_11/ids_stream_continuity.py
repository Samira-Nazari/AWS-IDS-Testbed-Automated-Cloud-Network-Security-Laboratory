"""Calculate continuous IDS window and buffer positions."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_WINDOW_SIZE = 30
DEFAULT_STRIDE = 5


@dataclass(frozen=True)
class ContinuityPlan:
    """Describe how available stream rows should be processed."""

    total_rows: int
    window_size: int
    stride: int
    window_count: int
    next_start_offset: int
    retained_rows: int

    @property
    def can_run_inference(self) -> bool:
        """Return whether the available rows can produce a model window."""
        return self.window_count > 0


def plan_continuous_windows(
    total_rows: int,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> ContinuityPlan:
    """Calculate new windows and the rows retained for the next CSV."""

    if total_rows < 0:
        raise ValueError("total_rows cannot be negative")

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    if total_rows < window_size:
        window_count = 0
    else:
        window_count = ((total_rows - window_size) // stride) + 1

    next_start_offset = window_count * stride
    retained_rows = total_rows - next_start_offset

    return ContinuityPlan(
        total_rows=total_rows,
        window_size=window_size,
        stride=stride,
        window_count=window_count,
        next_start_offset=next_start_offset,
        retained_rows=retained_rows,
    )
