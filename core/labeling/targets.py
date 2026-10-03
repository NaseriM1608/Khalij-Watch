"""Dense training targets for the stride-2 point detector, from index records.

For each chip: an objectness heatmap (a fixed-radius Gaussian per vessel, as
the xView3 winner used, which sidesteps unknown lengths), a loss weight (0
where the truth is unknown), and sub-pixel offsets at each vessel's peak.

What a chip teaches comes from its ``supervision`` (core.labeling.index);
the ``objects`` list places the other known detections inside it. Unlabelled
detections are never trained as sea: they may be vessels that broadcast
nothing, so their discs are drawn after the negative ones. Positives are drawn
last, so a vessel beside an unlabelled detection keeps its own supervision.

Geometry: the chip centre pixel is floor(row + 0.5, col + 0.5), rounding halves
up as core.labeling.chips cuts it, and sits at chip index 64;
chip pixel p maps to output coordinate (p - 0.5) / STRIDE, so output cell i
has its centre at chip pixel 2i + 0.5.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.inference.contract import CHIP_SIZE, OUTPUT_SIZE, STRIDE

# Output-pixel radii. HEAT_RADIUS is the fixed objectness radius (xView3 recipe).
HEAT_RADIUS = 3
# About 160 m: covers the hull of a large unidentified ship centred there.
IGNORE_RADIUS = 8
# The ground a centre negative vouches for.
CENTRE_RADIUS = 8
_SIGMA = (2 * HEAT_RADIUS + 1) / 6.0

_BASE_WEIGHT = {"positive": 1.0, "ignore": 1.0, "full_negative": 1.0, "centre_negative": 0.0}


def output_position(chip_coord: float) -> tuple[int, float]:
    """The output cell holding a chip coordinate, and the offset from that cell's centre."""
    out = (chip_coord - 0.5) / STRIDE
    cell = math.floor(out + 0.5)
    return cell, out - cell


def _centre_pixel(coord: float) -> int:
    """The scene pixel a chip is cut around: halves round up, as in core.labeling.chips."""
    return math.floor(coord + 0.5)


def encode_targets(record: dict[str, Any]) -> dict[str, np.ndarray]:
    """Heatmap, weight, offset (2, H, W) and offset_mask for one index record."""
    supervision = record["supervision"]
    if supervision not in _BASE_WEIGHT:
        raise ValueError(f"unknown supervision {supervision!r}")

    size = OUTPUT_SIZE
    heatmap = np.zeros((size, size), dtype=np.float32)
    weight = np.full((size, size), _BASE_WEIGHT[supervision], dtype=np.float32)
    offset = np.zeros((2, size, size), dtype=np.float32)
    offset_mask = np.zeros((size, size), dtype=np.float32)

    # Sub-pixel position of the chip centre: chips are cut around the centre pixel.
    half = CHIP_SIZE // 2
    centre_row = half + record["row"] - _centre_pixel(record["row"])
    centre_col = half + record["col"] - _centre_pixel(record["col"])
    centre = (supervision, centre_row, centre_col)
    placed = [centre] + [
        (label, centre_row + d_row, centre_col + d_col) for d_row, d_col, label in record["objects"]
    ]

    for label, row, col in placed:
        if label in ("centre_negative", "persistent"):
            _disc(weight, row, col, CENTRE_RADIUS, 1.0)
    for label, row, col in placed:
        if label in ("ignore", "unlabelled"):
            _disc(weight, row, col, IGNORE_RADIUS, 0.0)
    for label, row, col in placed:
        if label in ("positive", "vessel"):
            _draw_positive(heatmap, weight, offset, offset_mask, row, col)
    return {"heatmap": heatmap, "weight": weight, "offset": offset, "offset_mask": offset_mask}


def _draw_positive(
    heatmap: np.ndarray,
    weight: np.ndarray,
    offset: np.ndarray,
    offset_mask: np.ndarray,
    row: float,
    col: float,
) -> None:
    cell_row, off_row = output_position(row)
    cell_col, off_col = output_position(col)
    size = heatmap.shape[0]
    if not (0 <= cell_row < size and 0 <= cell_col < size):
        return
    rows, cols = np.ogrid[:size, :size]
    gaussian = np.exp(-((rows - cell_row) ** 2 + (cols - cell_col) ** 2) / (2 * _SIGMA**2))
    inside = (np.abs(rows - cell_row) <= HEAT_RADIUS) & (np.abs(cols - cell_col) <= HEAT_RADIUS)
    np.maximum(heatmap, np.where(inside, gaussian, 0.0), out=heatmap)
    weight[inside] = 1.0
    offset[:, cell_row, cell_col] = (off_row, off_col)
    offset_mask[cell_row, cell_col] = 1.0


def _disc(weight: np.ndarray, row: float, col: float, radius: float, value: float) -> None:
    """Set ``weight`` to ``value`` within ``radius`` output pixels of a chip position."""
    size = weight.shape[0]
    out_row = (row - 0.5) / STRIDE
    out_col = (col - 0.5) / STRIDE
    rows, cols = np.ogrid[:size, :size]
    weight[(rows - out_row) ** 2 + (cols - out_col) ** 2 <= radius**2] = value
