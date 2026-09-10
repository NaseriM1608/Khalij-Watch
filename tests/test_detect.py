"""Tests for the CFAR-to-candidate pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from core.sar.cfar import CfarConfig
from core.sar.detect import (
    MAX_LENGTH_M,
    PIXEL_SPACING_M,
    Candidate,
    detect_candidates,
    georeference,
    is_plausible_vessel,
    label_components,
    measure_component,
)
from core.sar.synthetic import Target, make_scene, ship_grid

# Defaults: the guard band must clear a vessel's LENGTH, not its beam.
CONFIG = CfarConfig(pfa=1e-6)


# --------------------------------------------------------------------------
# Connected components
# --------------------------------------------------------------------------


def test_separate_blobs_get_separate_labels():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True
    mask[12:15, 12:15] = True

    labels, count = label_components(mask)

    assert count == 2
    assert labels[3, 3] != labels[13, 13]


def test_diagonally_touching_pixels_are_one_component():
    """8-connectivity: a ship at 45 degrees is a diagonal chain of pixels."""
    mask = np.zeros((10, 10), dtype=bool)
    for i in range(5):
        mask[i + 2, i + 2] = True

    _, count = label_components(mask)

    assert count == 1


def test_an_empty_mask_has_no_components():
    labels, count = label_components(np.zeros((10, 10), dtype=bool))

    assert count == 0
    assert labels.sum() == 0


def test_a_u_shape_is_a_single_component():
    """Guards the union-find merge: two arms joined only at the bottom."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2] = True
    mask[2:8, 6] = True
    mask[7, 2:7] = True

    _, count = label_components(mask)

    assert count == 1


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_a_horizontal_object_measures_its_length_along_the_row():
    rows = np.full(21, 50)
    cols = np.arange(40, 61)

    length, width, _ = measure_component(rows, cols)

    assert length == pytest.approx(21 * PIXEL_SPACING_M, rel=0.1)
    assert width == pytest.approx(PIXEL_SPACING_M, rel=0.5)


def test_length_always_exceeds_width():
    rows, cols = np.meshgrid(np.arange(48, 53), np.arange(40, 61), indexing="ij")

    length, width, _ = measure_component(rows.ravel(), cols.ravel())

    assert length > width


def test_heading_is_reported_modulo_180():
    """A SAR return is symmetric: bow and stern cannot be distinguished."""
    rows = np.full(21, 50)
    cols = np.arange(40, 61)

    _, _, heading = measure_component(rows, cols)

    assert 0.0 <= heading < 180.0
    # An east-west object points along 90 degrees.
    assert heading == pytest.approx(90.0, abs=5.0)


def test_a_north_south_object_heads_roughly_north():
    rows = np.arange(40, 61)
    cols = np.full(21, 50)

    _, _, heading = measure_component(rows, cols)

    assert min(heading, 180.0 - heading) == pytest.approx(0.0, abs=5.0)


def test_a_single_pixel_has_no_meaningful_geometry():
    length, _width, heading = measure_component(np.array([5]), np.array([5]))

    assert length == PIXEL_SPACING_M
    assert heading == 0.0


# --------------------------------------------------------------------------
# Plausibility filter
# --------------------------------------------------------------------------


def test_a_realistic_vessel_passes():
    assert is_plausible_vessel(length_m=180.0, width_m=25.0, pixel_count=40)


def test_a_speckle_spike_is_rejected():
    assert not is_plausible_vessel(length_m=20.0, width_m=10.0, pixel_count=1)


def test_an_object_longer_than_any_ship_is_rejected():
    assert not is_plausible_vessel(length_m=MAX_LENGTH_M + 100, width_m=30.0, pixel_count=200)


def test_a_round_blob_is_rejected():
    """Ships are elongated; a circular return is more likely a platform."""
    assert not is_plausible_vessel(length_m=100.0, width_m=95.0, pixel_count=80)


def test_an_absurdly_thin_streak_is_rejected():
    """A one-pixel-wide streak across the scene is an artefact, not a ship."""
    assert not is_plausible_vessel(length_m=400.0, width_m=10.0, pixel_count=40)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_vessels_become_candidates():
    targets = ship_grid((500, 500), rows=2, cols=2, length_px=24, width_px=8)
    scene = make_scene((500, 500), targets=targets, seed=5)

    candidates = detect_candidates(scene.amplitude, config=CONFIG)

    assert len(candidates) == len(scene.targets)


def test_candidates_sit_on_their_targets():
    target = Target(250, 250, length_px=30, width_px=10)
    scene = make_scene((500, 500), targets=[target], seed=3)

    candidates = detect_candidates(scene.amplitude, config=CONFIG)

    assert len(candidates) == 1
    assert candidates[0].row == pytest.approx(250, abs=6)
    assert candidates[0].col == pytest.approx(250, abs=6)


def test_measured_length_approximates_the_true_length():
    """The length estimate is a first-class output, so it must be honest."""
    target = Target(250, 250, length_px=30, width_px=8)  # 300 m at 10 m/px
    scene = make_scene((500, 500), targets=[target], seed=4)

    candidates = detect_candidates(scene.amplitude, config=CONFIG)

    assert candidates[0].length_m == pytest.approx(300.0, rel=0.35)


def test_empty_sea_yields_no_candidates():
    scene = make_scene((500, 500), seed=8)

    assert detect_candidates(scene.amplitude, config=CONFIG) == []


def test_morphological_opening_removes_scattered_speckle():
    """The first line of defence, and it does most of the work.

    At a deliberately loose Pfa the threshold passes thousands of scattered
    clutter pixels, but speckle spikes are isolated and do not survive
    erosion. Measured on this scene: ~7,900 raw hits reduce to zero.
    """
    from core.sar.cfar import cfar_detect
    from core.sar.detect import _binary_opening

    scene = make_scene((400, 400), seed=12)
    hits = cfar_detect(scene.amplitude, CfarConfig(pfa=5e-2))

    assert hits.sum() > 1000
    assert _binary_opening(hits).sum() == 0


def test_the_geometry_filter_removes_survivors():
    """The last line: blobs that survive opening but cannot be ships.

    Pfa has to be pushed very loose to produce any, which is itself the
    point - by this stage there is little left to reject.
    """
    scene = make_scene((400, 400), seed=12)
    loose = CfarConfig(pfa=2e-1)

    unfiltered = detect_candidates(scene.amplitude, config=loose, apply_geometry_filter=False)
    filtered = detect_candidates(scene.amplitude, config=loose)

    assert len(unfiltered) > 10
    assert len(filtered) < len(unfiltered) / 10


def test_masked_land_produces_no_candidates():
    scene = make_scene((400, 400), land_rows=120, seed=4)

    candidates = detect_candidates(scene.amplitude, config=CONFIG, mask=scene.land)

    assert all(c.row >= 120 for c in candidates)


def test_a_vessel_just_offshore_survives_the_land_mask():
    coast = 120
    ship = Target(coast + 30, 200, length_px=26, width_px=8)
    scene = make_scene((400, 400), targets=[ship], land_rows=coast, seed=6)

    candidates = detect_candidates(scene.amplitude, config=CONFIG, mask=scene.land)

    assert any(abs(c.row - ship.row) < 10 for c in candidates)


# --------------------------------------------------------------------------
# Georeferencing
# --------------------------------------------------------------------------


def test_georeferencing_places_candidates_on_the_earth():
    from rasterio.transform import from_bounds

    transform = from_bounds(56.0, 26.0, 57.0, 27.0, 1000, 1000)
    candidate = Candidate(
        row=500.0,
        col=500.0,
        length_m=200.0,
        width_m=30.0,
        heading_deg=45.0,
        pixel_count=40,
        peak_amplitude=1500.0,
        mean_amplitude=900.0,
    )

    located = georeference([candidate], transform)[0]

    assert located.lon == pytest.approx(56.5, abs=0.01)
    assert located.lat == pytest.approx(26.5, abs=0.01)


def test_a_window_offset_maps_back_to_scene_coordinates():
    """A candidate found in a tile must land at its true scene position."""
    from rasterio.transform import from_bounds

    transform = from_bounds(56.0, 26.0, 57.0, 27.0, 1000, 1000)
    candidate = Candidate(
        row=10.0,
        col=10.0,
        length_m=200.0,
        width_m=30.0,
        heading_deg=0.0,
        pixel_count=40,
        peak_amplitude=1500.0,
        mean_amplitude=900.0,
    )

    located = georeference([candidate], transform, window_offset=(490, 490))[0]

    assert located.row == 500.0
    assert located.lon == pytest.approx(56.5, abs=0.01)


def test_aspect_ratio_is_exposed():
    candidate = Candidate(
        row=0.0,
        col=0.0,
        length_m=200.0,
        width_m=25.0,
        heading_deg=0.0,
        pixel_count=40,
        peak_amplitude=1.0,
        mean_amplitude=1.0,
    )

    assert candidate.aspect_ratio == pytest.approx(8.0)


def test_a_guard_band_shorter_than_the_vessel_suppresses_it():
    """The quietest CFAR failure, pinned.

    If the guard box is shorter than the ship, the ship's own bright pixels
    fall into its training ring, inflate its local background, and suppress
    its own detection. The symptom is a large vessel appearing as a few
    fragments around its edges - or a badly short length estimate - rather
    than an outright miss, so it is easy to overlook.
    """
    target = Target(300, 300, length_px=30, width_px=10)  # 300 m
    scene = make_scene((600, 600), targets=[target], seed=3)

    too_small = CfarConfig(guard_half=6, train_half=14, pfa=1e-6)  # 130 m guard
    adequate = CfarConfig(pfa=1e-6)  # 490 m guard

    starved = detect_candidates(scene.amplitude, config=too_small)
    healthy = detect_candidates(scene.amplitude, config=adequate)

    assert healthy[0].pixel_count > starved[0].pixel_count * 5
    assert healthy[0].length_m == pytest.approx(300.0, rel=0.15)
    assert starved[0].length_m < 200.0


def test_length_is_recovered_across_the_vessel_size_range():
    """Length is a first-class output, so accuracy must hold at both ends."""
    for length_px, tolerance in ((30, 0.15), (12, 0.25), (4, 0.5)):
        target = Target(300, 300, length_px=length_px, width_px=max(3, length_px // 3))
        scene = make_scene((600, 600), targets=[target], seed=3)

        candidates = detect_candidates(scene.amplitude, config=CONFIG)

        assert candidates, f"missed a {length_px * 10} m vessel"
        assert candidates[0].length_m == pytest.approx(length_px * 10.0, rel=tolerance)
