"""Vessel extent must be measured from the hull, not the sidelobe cross.

On a real Sentinel-1 VH window over the Fujairah anchorage (2026-09-10), the
CFAR mask of bright vessels included their sidelobes and gave a median width
of 135 m; no ship is that broad. Measuring only pixels within 12 dB of each
target's peak gave 62 m. These tests reproduce the effect on a synthetic
target so the correction stays in place.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.sar.cfar import CfarConfig
from core.sar.detect import EXTENT_PEAK_FRACTION, detect_candidates, hull_pixels
from core.sar.synthetic import Target, make_scene

CONFIG = CfarConfig(pfa=1e-6)
CENTRE = 300
# 21 x 5 pixels at 10 m: a 210 m by 50 m vessel lying along the columns.
VESSEL = Target(CENTRE, CENTRE, length_px=20, width_px=5, amplitude=1400.0)
# Bright enough to clear the CFAR threshold over sea, but below 25% of peak.
SIDELOBE_DN = 300.0
ARM_PX = 30
ARM_HALF_THICKNESS = 1
# Brighter than SIDELOBE_DN so a one-sided streak drags the raw centroid measurably,
# yet still below 25% of the vessel's ~1897 DN peak (474 DN).
BLEED_DN = 450.0


def with_sidelobes(amplitude: np.ndarray, *, left: bool = True, right: bool = True) -> np.ndarray:
    """Add a sidelobe cross, three pixels thick, centred on the vessel."""
    out = amplitude.copy()
    band = slice(CENTRE - ARM_HALF_THICKNESS, CENTRE + ARM_HALF_THICKNESS + 1)
    col_start = CENTRE - ARM_PX if left else CENTRE
    col_stop = CENTRE + ARM_PX + 1 if right else CENTRE + 1
    out[band, col_start:col_stop] = np.maximum(out[band, col_start:col_stop], SIDELOBE_DN)
    column = slice(CENTRE - ARM_PX, CENTRE + ARM_PX + 1)
    out[column, band] = np.maximum(out[column, band], SIDELOBE_DN)
    return out


@pytest.fixture(scope="module")
def cross_scene() -> np.ndarray:
    scene = make_scene((600, 600), targets=[VESSEL], seed=21)
    return with_sidelobes(scene.amplitude)


def test_the_raw_cfar_mask_measures_the_sidelobe_cross(cross_scene):
    """Establishes the failure the correction exists for."""
    raw = detect_candidates(
        cross_scene, config=CONFIG, apply_geometry_filter=False, extent_peak_fraction=0.0
    )
    vessel = max(raw, key=lambda c: c.pixel_count)

    # The arms make the mask roughly as tall as it is long.
    assert vessel.width_m > 300.0


def test_hull_measurement_recovers_the_vessel_geometry(cross_scene):
    candidates = detect_candidates(cross_scene, config=CONFIG)

    assert len(candidates) == 1
    vessel = candidates[0]
    assert vessel.width_m < 100.0
    assert vessel.length_m == pytest.approx(210.0, rel=0.3)


def test_the_hull_passes_the_plausibility_filter_where_the_cross_would_not(cross_scene):
    """A near-square cross fails the aspect-ratio test, so a real ship would be lost."""
    raw = detect_candidates(cross_scene, config=CONFIG, extent_peak_fraction=0.0)
    hull = detect_candidates(cross_scene, config=CONFIG)

    assert len(raw) < len(hull)


def with_one_sided_bleed(amplitude: np.ndarray, length_px: int = 50) -> np.ndarray:
    """Bleed along the range axis on one side of the vessel only, three pixels thick."""
    out = amplitude.copy()
    band = slice(CENTRE - ARM_HALF_THICKNESS, CENTRE + ARM_HALF_THICKNESS + 1)
    start = CENTRE + VESSEL.length_px // 2 + 1
    stop = start + length_px
    out[band, start:stop] = np.maximum(out[band, start:stop], BLEED_DN)
    return out


def nearest_to_centre(candidates):
    """The vessel, not a detached streak that may hold more pixels."""
    return min(candidates, key=lambda c: abs(c.row - CENTRE) + abs(c.col - CENTRE))


def test_one_sided_bleed_does_not_drag_the_position():
    """Position feeds AIS matching, so asymmetric bleed must not shift it."""
    scene = make_scene((600, 600), targets=[VESSEL], seed=21)
    lopsided = with_one_sided_bleed(scene.amplitude)

    raw = detect_candidates(
        lopsided, config=CONFIG, apply_geometry_filter=False, extent_peak_fraction=0.0
    )
    hull = detect_candidates(lopsided, config=CONFIG, apply_geometry_filter=False)
    raw_vessel = nearest_to_centre(raw)
    hull_vessel = nearest_to_centre(hull)

    assert raw_vessel.col - CENTRE > 5.0
    assert abs(hull_vessel.col - CENTRE) < 2.0


def with_bright_core(amplitude: np.ndarray, core_dn: float = 10_000.0) -> np.ndarray:
    """A 3x3 superstructure-like core far brighter than the rest of the hull."""
    out = amplitude.copy()
    out[CENTRE - 1 : CENTRE + 2, CENTRE - 1 : CENTRE + 2] = core_dn
    return out


def test_a_vessel_with_a_dominant_bright_core_is_still_reported():
    """Regression from the real Fujairah VH window: three of the brightest
    targets (peaks 1851-2460 DN) were dropped. Within 12 dB of so bright a peak
    only a near-square core remains, which fails the aspect test, although the
    whole component has a ship's shape."""
    scene = make_scene((600, 600), targets=[VESSEL], seed=21)

    candidates = detect_candidates(with_bright_core(scene.amplitude), config=CONFIG)

    assert len(candidates) == 1
    vessel = candidates[0]
    assert vessel.length_m == pytest.approx(210.0, rel=0.3)
    # Position still comes from the bright core, not the whole component.
    assert abs(vessel.col - CENTRE) < 2.0
    assert abs(vessel.row - CENTRE) < 2.0


def test_a_compact_square_blob_is_still_rejected():
    """The whole-component check must not admit what the aspect test exists for."""
    scene = make_scene((600, 600), seed=21)
    blob = scene.amplitude.copy()
    blob[CENTRE - 3 : CENTRE + 4, CENTRE - 3 : CENTRE + 4] = 1400.0

    assert detect_candidates(blob, config=CONFIG) == []


def test_hull_pixels_keep_only_the_bright_core():
    rows = np.arange(6)
    cols = np.arange(6)
    values = np.array([1000.0, 900.0, 800.0, 200.0, 150.0, 100.0])

    hull_rows, hull_cols = hull_pixels(rows, cols, values, fraction=0.25)

    assert hull_rows.tolist() == [0, 1, 2]
    assert hull_cols.tolist() == [0, 1, 2]


def test_hull_pixels_fall_back_to_the_whole_component_when_too_few_survive():
    rows = np.arange(4)
    cols = np.arange(4)
    values = np.array([1000.0, 10.0, 10.0, 10.0])

    hull_rows, _ = hull_pixels(rows, cols, values, fraction=0.25)

    assert hull_rows.tolist() == [0, 1, 2, 3]


def test_a_zero_fraction_measures_the_raw_mask():
    rows = np.arange(3)
    values = np.array([1000.0, 1.0, 1.0])

    hull_rows, _ = hull_pixels(rows, rows, values, fraction=0.0)

    assert hull_rows.tolist() == [0, 1, 2]


def test_the_default_threshold_is_twelve_decibels():
    """0.25 in amplitude is -12 dB; changing it changes every length reported."""
    assert 20.0 * np.log10(EXTENT_PEAK_FRACTION) == pytest.approx(-12.04, abs=0.01)
