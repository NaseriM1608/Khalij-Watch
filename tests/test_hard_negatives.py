"""Tests for hard negative sampling: land edges, bright sea clutter, platforms."""

from __future__ import annotations

import numpy as np
import pytest

from core.labeling.chips import CHIP_SIZE
from core.labeling.hard_negatives import (
    chips_inside,
    clutter_centres,
    coast_edge_centres,
)

HALF = CHIP_SIZE // 2


def land_west_of(col: int, shape: tuple[int, int] = (600, 600)) -> np.ndarray:
    land = np.zeros(shape, dtype=bool)
    land[:, :col] = True
    return land


def sea(shape: tuple[int, int] = (600, 600), level: float = 20.0, seed: int = 0) -> np.ndarray:
    """Speckled amplitude around ``level``, like VH open sea."""
    rng = np.random.default_rng(seed)
    return np.sqrt(rng.gamma(4.4, level**2 / 4.4, size=shape))


# --- coast edges -------------------------------------------------------------


def test_coast_centres_sit_on_land_pixels_next_to_water():
    land = land_west_of(300)
    nodata = np.zeros_like(land)

    centres = coast_edge_centres(land, nodata, [], count=5, rng=np.random.default_rng(1))

    assert centres
    for row, col in centres:
        r, c = int(row), int(col)
        assert land[r, c]
        assert not land[r, c + 1]


def test_coast_chips_fit_in_the_window_and_do_not_overlap():
    land = land_west_of(300)

    centres = coast_edge_centres(
        land, np.zeros_like(land), [], count=10, rng=np.random.default_rng(2)
    )

    for row, col in centres:
        assert HALF <= row <= land.shape[0] - HALF
        assert HALF <= col <= land.shape[1] - HALF
    for i, (r1, c1) in enumerate(centres):
        for r2, c2 in centres[i + 1 :]:
            assert abs(r1 - r2) >= CHIP_SIZE or abs(c1 - c2) >= CHIP_SIZE


def test_coast_centres_keep_clear_of_detections():
    """A vessel moored off the coast must not sit inside a land-edge negative."""
    land = land_west_of(300)
    targets = [(float(r), 330.0) for r in range(0, 600, 20)]

    centres = coast_edge_centres(
        land, np.zeros_like(land), targets, count=5, rng=np.random.default_rng(3), clearance_px=40
    )

    assert centres == []


def test_coast_chips_avoid_nodata():
    land = land_west_of(300)
    nodata = np.zeros_like(land)
    nodata[:, 350:] = True

    centres = coast_edge_centres(land, nodata, [], count=5, rng=np.random.default_rng(4))

    assert centres == []


def test_no_coast_no_centres():
    land = np.zeros((600, 600), dtype=bool)

    assert coast_edge_centres(land, land, [], count=5, rng=np.random.default_rng(5)) == []


def test_mostly_land_chips_are_not_coast_negatives():
    """A pond-sized patch of water inland is not a coastline worth a chip."""
    land = np.ones((600, 600), dtype=bool)
    land[295:305, 295:305] = False

    centres = coast_edge_centres(
        land, np.zeros_like(land), [], count=5, rng=np.random.default_rng(6)
    )

    assert centres == []


# --- bright sea clutter --------------------------------------------------------


def test_clutter_centres_land_in_the_bright_patch():
    band = sea()
    band[:300, :300] *= 2.0  # a wind-roughened patch
    masked = np.zeros(band.shape, dtype=bool)

    centres = clutter_centres(band, masked, [], count=4, rng=np.random.default_rng(7))

    assert centres
    for row, col in centres:
        # Mostly inside the patch: a chip straddling its edge may still qualify.
        assert row < 300 + HALF and col < 300 + HALF


def test_clutter_skips_windows_holding_a_point_target():
    """A bright spot is a possible vessel that CFAR missed: never a negative."""
    band = sea()
    band[:300, :300] *= 2.0
    for r in range(10, 300, 30):
        for c in range(10, 300, 30):
            band[r, c] = 2000.0
    masked = np.zeros(band.shape, dtype=bool)

    centres = clutter_centres(band, masked, [], count=4, rng=np.random.default_rng(8))

    assert centres == []


def test_uniform_sea_has_no_bright_clutter():
    band = sea()
    masked = np.zeros(band.shape, dtype=bool)

    assert clutter_centres(band, masked, [], count=4, rng=np.random.default_rng(9)) == []


def test_clutter_respects_mask_and_clearance():
    band = sea()
    band[:300, :300] *= 2.0
    masked = np.zeros(band.shape, dtype=bool)
    masked[:300, :150] = True
    targets = [(150.0, 225.0)]

    centres = clutter_centres(
        band, masked, targets, count=4, rng=np.random.default_rng(10), clearance_px=200
    )

    assert centres == []


# --- platforms and other fixed points ---------------------------------------------


def test_chips_inside_keeps_points_whose_chip_fits():
    points = [(100.0, 100.0), (10.0, 300.0), (300.0, 590.0), (536.0, 64.0)]

    assert chips_inside(points, (600, 600)) == [(100.0, 100.0), (536.0, 64.0)]


@pytest.mark.parametrize("count", [0, -1])
def test_nothing_requested_nothing_returned(count):
    band = sea()
    band[:300, :300] *= 2.0
    masked = np.zeros(band.shape, dtype=bool)

    assert clutter_centres(band, masked, [], count=count, rng=np.random.default_rng(0)) == []
    assert (
        coast_edge_centres(land_west_of(300), masked, [], count=count, rng=np.random.default_rng(0))
        == []
    )
