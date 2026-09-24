"""Cut fixed-size chips out of a SAR window for the training dataset.

Chips keep raw amplitude as uint16. The frozen sigmoid in core.sar.windows was
calibrated on VV, where open sea sits near 80 DN. On VH the sea is nearer
18 DN and many ships peak at a few hundred, so baking that curve into 8-bit
chips would fix a VV calibration into the VH channel. Normalisation is chosen
and frozen at training time instead, at twice the storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CHIP_SIZE = 128
BAND_ORDER = ("vv", "vh")

# A chip with more than this share of nodata in any band straddles the swath edge.
MAX_NODATA_FRACTION = 0.1

_UINT16_MAX = 65_535


@dataclass(frozen=True)
class ChipRequest:
    """Where to cut a chip: the scene pixel coordinates of its centre."""

    chip_id: str
    row: float
    col: float


def extract_chips(
    bands: np.ndarray,
    requests: Sequence[ChipRequest],
    *,
    window_offset: tuple[int, int] = (0, 0),
    size: int = CHIP_SIZE,
) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """Cut a chip around each request from a (band, row, col) window stack.

    The requested pixel lands at index ``size // 2`` of the chip. Returns the
    chips as (N, band, size, size) uint16, the ids kept in order, and an
    ``(id, reason)`` pair for every request skipped.
    """
    stack = np.asarray(bands)
    if stack.ndim != 3:
        raise ValueError(f"bands must be a (band, row, col) stack, got shape {stack.shape}")
    band_count, height, width = stack.shape
    half = size // 2
    row_off, col_off = window_offset

    chips: list[np.ndarray] = []
    kept: list[str] = []
    skipped: list[tuple[str, str]] = []
    for request in requests:
        row = int(np.floor(request.row - row_off + 0.5))
        col = int(np.floor(request.col - col_off + 0.5))
        top, left = row - half, col - half
        if top < 0 or left < 0 or top + size > height or left + size > width:
            skipped.append((request.chip_id, "outside_window"))
            continue
        chip = stack[:, top : top + size, left : left + size]
        if ((chip == 0).mean(axis=(1, 2)) > MAX_NODATA_FRACTION).any():
            skipped.append((request.chip_id, "nodata"))
            continue
        chips.append(_as_uint16(chip))
        kept.append(request.chip_id)

    if not chips:
        return np.zeros((0, band_count, size, size), dtype=np.uint16), kept, skipped
    return np.stack(chips), kept, skipped


def _as_uint16(chip: np.ndarray) -> np.ndarray:
    if chip.dtype == np.uint16:
        return chip.copy()
    return np.clip(np.rint(chip), 0, _UINT16_MAX).astype(np.uint16)
