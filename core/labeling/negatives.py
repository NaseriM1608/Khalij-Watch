"""Open-sea negative chips: water with nothing in it that anyone detected.

Negatives are mandatory; without them a model learns that every bright pixel
is a ship. This module supplies the simplest kind, empty sea. The clearance
from targets matters: an unlabelled CFAR detection may be a vessel that
broadcasts nothing, so no negative chip may come near one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.labeling.chips import CHIP_SIZE

# 1 km at 10 m pixels, clear of any detection or label.
DEFAULT_CLEARANCE_PX = 100
# Random draws per requested point before giving up on a crowded window.
ATTEMPTS_PER_POINT = 50


def sample_sea_negatives(
    masked: np.ndarray,
    targets: Sequence[tuple[float, float]],
    *,
    count: int,
    rng: np.random.Generator,
    clearance_px: float = DEFAULT_CLEARANCE_PX,
    size: int = CHIP_SIZE,
) -> list[tuple[float, float]]:
    """Window (row, col) chip centres on unmasked sea, clear of every target.

    ``masked`` marks land, platforms and nodata. ``targets`` holds the window
    pixel position of every detection and label. Every returned chip lies
    wholly on unmasked pixels, and returned chips do not overlap each other.
    Fewer than ``count`` points, possibly none, come back when the window
    cannot supply them.
    """
    height, width = masked.shape
    half = size // 2
    if count <= 0 or height < size or width < size:
        return []

    blocked = _blocked_chip_windows(masked, size)
    attempts = count * ATTEMPTS_PER_POINT
    rows = rng.integers(half, height - half + 1, size=attempts)
    cols = rng.integers(half, width - half + 1, size=attempts)
    free = ~blocked[rows - half, cols - half]

    target_array = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    if len(target_array):
        distance = np.hypot(
            rows[:, None] - target_array[None, :, 0], cols[:, None] - target_array[None, :, 1]
        ).min(axis=1)
        free &= distance >= clearance_px

    accepted: list[tuple[float, float]] = []
    for row, col in zip(rows[free], cols[free], strict=True):
        if all(abs(row - r) >= size or abs(col - c) >= size for r, c in accepted):
            accepted.append((float(row), float(col)))
            if len(accepted) == count:
                break
    return accepted


def _blocked_chip_windows(masked: np.ndarray, size: int) -> np.ndarray:
    """For each top-left corner, whether its size x size window holds a masked pixel."""
    integral = np.zeros((masked.shape[0] + 1, masked.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = masked.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    counts = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return counts > 0
