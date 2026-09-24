"""Visual QA contact sheets for chip datasets.

Most labelling bugs - a sign error in an offset, a swapped axis, a wrong
scene - are obvious to the eye and invisible in metrics. Every dataset build
therefore renders its chips in a grid with the centre marked, so a person can
check that the ships sit where the labels put them.
"""

from __future__ import annotations

import math

import numpy as np

from core.sar.display import display_stretch

GAP_PX = 2
BACKGROUND = 24
# Ticks mark the centre from a short distance, leaving the target itself visible.
TICK_NEAR_PX = 6
TICK_FAR_PX = 10
TICK_COLOUR = (255, 40, 40)
# VH, where vessels stand out best against the darker sea.
DEFAULT_CHANNEL = 1


def contact_sheet(
    chips: np.ndarray, *, columns: int = 8, channel: int = DEFAULT_CHANNEL
) -> np.ndarray:
    """Tile (N, band, size, size) chips into one RGB image, (3, height, width) uint8.

    Each chip is stretched on its own, so a faint vessel stays visible beside
    a bright one.
    """
    if len(chips) == 0:
        raise ValueError("no chips to show")
    count, _, size, _ = chips.shape
    columns = max(1, min(columns, count))
    rows = math.ceil(count / columns)
    sheet = np.full(
        (3, rows * size + (rows + 1) * GAP_PX, columns * size + (columns + 1) * GAP_PX),
        BACKGROUND,
        dtype=np.uint8,
    )
    for index, chip in enumerate(chips):
        top = GAP_PX + (index // columns) * (size + GAP_PX)
        left = GAP_PX + (index % columns) * (size + GAP_PX)
        sheet[:, top : top + size, left : left + size] = display_stretch(chip[channel])
        _mark_centre(sheet, top + size // 2, left + size // 2)
    return sheet


def _mark_centre(sheet: np.ndarray, row: int, col: int) -> None:
    for distance in range(TICK_NEAR_PX, TICK_FAR_PX + 1):
        for r, c in (
            (row, col - distance),
            (row, col + distance),
            (row - distance, col),
            (row + distance, col),
        ):
            sheet[:, r, c] = TICK_COLOUR
