from __future__ import annotations

from typing import Optional, Tuple


def clamp_monotonic_time(
    prev_s: Optional[float],
    current_s: Optional[float],
    epsilon_s: float = 1e-6,
) -> Tuple[Optional[float], bool]:
    if current_s is None:
        return None, False
    if prev_s is None:
        return current_s, False
    if current_s + epsilon_s < prev_s:
        return prev_s, True
    return current_s, False
