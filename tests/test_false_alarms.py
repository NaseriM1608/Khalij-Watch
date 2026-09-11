"""Measuring CFAR false alarms where the truth is unknown.

Real imagery has no truth map, so the measurement counts CFAR hits on sea
clutter away from every object the detector found. These tests check, on
synthetic scenes where the truth IS known, that this count recovers the
requested Pfa and is not inflated by vessels or masked pixels.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.sar.cfar import CfarConfig, cfar_detect
from core.sar.false_alarms import PfaMeasurement, dilate, measure_clutter_false_alarms
from core.sar.synthetic import make_scene, ship_grid

SHAPE = (1200, 1200)


def test_dilation_grows_a_pixel_into_a_square():
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True

    grown = dilate(mask, 2)

    assert grown.sum() == 25
    assert grown[3:8, 3:8].all()


def test_dilation_is_clipped_at_the_image_edge():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True

    assert dilate(mask, 1).sum() == 4


def test_zero_radius_dilation_changes_nothing():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 3] = True

    assert np.array_equal(dilate(mask, 0), mask)


def test_measured_rate_matches_the_request_on_ideal_clutter():
    """On Gamma speckle the CFAR dial is exact, so the measurement must be too."""
    scene = make_scene(SHAPE, seed=3)

    (row,) = measure_clutter_false_alarms(scene.amplitude, pfas=(1e-3,))

    assert row.measured_pfa == pytest.approx(1e-3, rel=0.15)


def test_vessels_are_not_booked_as_false_alarms():
    scene = make_scene(SHAPE, targets=ship_grid(SHAPE), seed=4)

    (row,) = measure_clutter_false_alarms(scene.amplitude, pfas=(1e-3,))
    naive = float(cfar_detect(scene.amplitude, CfarConfig(pfa=1e-3)).mean())

    assert row.measured_pfa == pytest.approx(1e-3, rel=0.15)
    # Counting every hit, as on a target-free calibration scene, books each
    # vessel pixel as a false alarm.
    assert naive > 1.3e-3


def test_masked_pixels_are_neither_hits_nor_clutter():
    scene = make_scene(SHAPE, seed=5)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, :600] = True

    (row,) = measure_clutter_false_alarms(scene.amplitude, mask=mask, pfas=(1e-3,))

    assert row.clutter_pixels <= 600 * 1200
    assert row.measured_pfa == pytest.approx(1e-3, rel=0.2)


def test_every_setting_shares_one_denominator_and_reports_its_candidates():
    """The clutter area is fixed by the reference setting, so rows compare."""
    scene = make_scene(SHAPE, targets=ship_grid(SHAPE), seed=6)

    rows = measure_clutter_false_alarms(scene.amplitude, pfas=(1e-6, 1e-5))

    assert [r.requested_pfa for r in rows] == [1e-6, 1e-5]
    assert rows[0].clutter_pixels == rows[1].clutter_pixels
    assert rows[0].candidates == 9


def test_rate_and_excess_arithmetic():
    row = PfaMeasurement(
        requested_pfa=1e-4, alpha=3.0, clutter_pixels=1_000_000, clutter_hits=300, candidates=2
    )

    assert row.measured_pfa == pytest.approx(3e-4)
    assert row.excess == pytest.approx(3.0)


def test_no_clutter_pixels_gives_nan_rather_than_a_crash():
    row = PfaMeasurement(
        requested_pfa=1e-4, alpha=3.0, clutter_pixels=0, clutter_hits=0, candidates=0
    )

    assert math.isnan(row.measured_pfa)
    assert math.isnan(row.excess)
