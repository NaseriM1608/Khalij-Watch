"""Hard negative chips: things a naive detector calls a ship that are not ships.

PLAN.md section 8 lists the sources. Measured on our windows (PLAN.md section
17, 2026-09-29): offshore platforms and land edges are real hazards, while
azimuth ghosts of bright ships are not visible at 10 m and are not mined.
Bright, wind-roughened sea is taken only where no point target stands in the
chip, because a bright spot there may be a vessel that CFAR missed; for the
same reason every chip keeps clear of every detection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from core.labeling.chips import CHIP_SIZE

# 300 m: vessels moored along a coast sit just outside the 150 m coast buffer.
COAST_CLEARANCE_PX = 30
# 1 km, as for open-sea negatives.
CLUTTER_CLEARANCE_PX = 100
# A land-edge chip is a real coastline: some land, mostly not land.
MIN_LAND_FRACTION = 0.1
MAX_LAND_FRACTION = 0.7
# Chip mean over the window's sea median: how much brighter than calm sea.
CLUTTER_MIN_RATIO = 1.3
# Brightest pixel over the chip median: above this a point target may be present.
CLUTTER_MAX_PEAK_RATIO = 4.0
# Random candidates examined per requested chip before giving up.
CANDIDATES_PER_POINT = 50


def chips_inside(
    points: Sequence[tuple[float, float]], shape: tuple[int, int], *, size: int = CHIP_SIZE
) -> list[tuple[float, float]]:
    """The window (row, col) points whose chip lies wholly inside the window."""
    half = size // 2
    height, width = shape
    return [(r, c) for r, c in points if half <= r <= height - half and half <= c <= width - half]


def coast_edge_centres(
    land: np.ndarray,
    nodata: np.ndarray,
    targets: Sequence[tuple[float, float]],
    *,
    count: int,
    rng: np.random.Generator,
    clearance_px: float = COAST_CLEARANCE_PX,
    size: int = CHIP_SIZE,
    min_land: float = MIN_LAND_FRACTION,
    max_land: float = MAX_LAND_FRACTION,
) -> list[tuple[float, float]]:
    """Window (row, col) centres on land pixels that touch water.

    Each chip holds between ``min_land`` and ``max_land`` land, no nodata,
    and keeps ``clearance_px`` from every target; chips do not overlap.
    """
    land = np.asarray(land, dtype=bool)
    height, width = land.shape
    if count <= 0 or height < size or width < size or not land.any():
        return []
    half = size // 2

    water = ~land
    touches_water = np.zeros_like(land)
    touches_water[1:, :] |= water[:-1, :]
    touches_water[:-1, :] |= water[1:, :]
    touches_water[:, 1:] |= water[:, :-1]
    touches_water[:, :-1] |= water[:, 1:]
    edge = land & touches_water

    land_fraction = window_sums(land, size) / float(size * size)
    usable = (
        (land_fraction >= min_land)
        & (land_fraction <= max_land)
        & (window_sums(np.asarray(nodata, dtype=bool), size) == 0)
    )
    top, left = np.nonzero(edge[half : height - half + 1, half : width - half + 1] & usable)
    return _pick(top + half, left + half, targets, clearance_px, count, rng, size)


def clutter_centres(
    band: np.ndarray,
    masked: np.ndarray,
    targets: Sequence[tuple[float, float]],
    *,
    count: int,
    rng: np.random.Generator,
    clearance_px: float = CLUTTER_CLEARANCE_PX,
    size: int = CHIP_SIZE,
    min_ratio: float = CLUTTER_MIN_RATIO,
    max_peak_ratio: float = CLUTTER_MAX_PEAK_RATIO,
) -> list[tuple[float, float]]:
    """Window (row, col) centres of unmasked sea chips well above calm-sea brightness.

    A chip qualifies when its mean reaches ``min_ratio`` times the median of
    the window's unmasked pixels and its brightest pixel stays within
    ``max_peak_ratio`` times its own median. Chips keep ``clearance_px`` from
    every target and do not overlap.
    """
    values = np.asarray(band, dtype=np.float64)
    masked = np.asarray(masked, dtype=bool)
    height, width = values.shape
    if count <= 0 or height < size or width < size or masked.all():
        return []
    half = size // 2

    sea_level = float(np.median(values[~masked]))
    means = window_sums(np.where(masked, 0.0, values), size) / float(size * size)
    bright = (window_sums(masked, size) == 0) & (means >= min_ratio * sea_level)
    top, left = np.nonzero(bright)

    def calm(row: int, col: int) -> bool:
        chip = values[row - half : row + half, col - half : col + half]
        return bool(chip.max() <= max_peak_ratio * np.median(chip))

    return _pick(top + half, left + half, targets, clearance_px, count, rng, size, accept=calm)


def window_sums(values: np.ndarray, size: int) -> np.ndarray:
    """Sum over every size x size window, indexed by the window's top-left pixel."""
    array = np.asarray(values, dtype=np.float64)
    integral = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = array.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def _pick(
    rows: np.ndarray,
    cols: np.ndarray,
    targets: Sequence[tuple[float, float]],
    clearance_px: float,
    count: int,
    rng: np.random.Generator,
    size: int,
    accept: Callable[[int, int], bool] | None = None,
) -> list[tuple[float, float]]:
    """Randomly chosen candidates clear of targets, not overlapping, passing ``accept``."""
    if len(rows) == 0:
        return []
    chosen = rng.choice(len(rows), size=min(len(rows), count * CANDIDATES_PER_POINT), replace=False)
    rows, cols = rows[chosen], cols[chosen]

    target_array = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    if len(target_array):
        distance = np.hypot(
            rows[:, None] - target_array[None, :, 0], cols[:, None] - target_array[None, :, 1]
        ).min(axis=1)
        clear = distance >= clearance_px
        rows, cols = rows[clear], cols[clear]

    accepted: list[tuple[float, float]] = []
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if any(abs(row - r) < size and abs(col - c) < size for r, c in accepted):
            continue
        if accept is not None and not accept(row, col):
            continue
        accepted.append((float(row), float(col)))
        if len(accepted) == count:
            break
    return accepted
