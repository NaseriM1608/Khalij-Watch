"""CFAR detector tests against synthetic SAR with known ground truth.

The headline property is that Pfa is a DIAL, not an emergent accident: asking
for a false-alarm rate should produce roughly that rate on target-free sea.
That can only be checked where the truth is constructed.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.sar.cfar import (
    CfarConfig,
    cfar_detect,
    clutter_mean,
    measured_false_alarm_rate,
)
from core.sar.synthetic import (
    IW_GRDH_LOOKS,
    SEA_AMPLITUDE,
    Target,
    make_scene,
    ship_grid,
    speckle,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_training_cells_are_the_ring_between_the_boxes():
    config = CfarConfig(guard_half=2, train_half=4)

    # 9x9 outer minus 5x5 guard.
    assert config.outer_size == 9
    assert config.guard_size == 5
    assert config.training_cells == 81 - 25


def test_guard_must_be_smaller_than_the_training_window():
    """Otherwise there is no ring left to estimate clutter from."""
    with pytest.raises(ValueError, match="must exceed"):
        CfarConfig(guard_half=8, train_half=8)


def test_pfa_must_be_a_probability():
    with pytest.raises(ValueError, match="pfa"):
        CfarConfig(pfa=0.0)
    with pytest.raises(ValueError, match="pfa"):
        CfarConfig(pfa=1.5)


def test_alpha_rises_as_the_false_alarm_target_tightens():
    """A stricter Pfa demands a higher threshold."""
    loose = CfarConfig(pfa=1e-3).alpha
    strict = CfarConfig(pfa=1e-8).alpha

    assert strict > loose > 1.0


def test_a_tiny_training_ring_is_rejected():
    """8 cells is arithmetically valid but statistically useless."""
    with pytest.raises(ValueError, match="training cells"):
        CfarConfig(guard_half=0, train_half=1)


def test_single_look_threshold_matches_the_textbook_formula():
    """For L=1 the Gamma model must reduce to alpha = -ln(Pfa)."""
    import math

    from core.sar.cfar import gamma_threshold

    for pfa in (1e-3, 1e-6, 1e-9):
        assert gamma_threshold(1.0, pfa) == pytest.approx(-math.log(pfa), rel=1e-6)


def test_multilook_threshold_is_well_below_the_single_look_one():
    """The trap this guards against.

    Sentinel-1 GRDH is ~4.4 looks. Using the single-look formula sets the
    threshold roughly 2.7x too high, so the detector runs far more
    conservatively than its Pfa dial claims and quietly misses vessels.
    """
    from core.sar.cfar import gamma_threshold

    single = gamma_threshold(1.0, 1e-6)
    multi = gamma_threshold(4.4, 1e-6)

    assert multi < single / 2
    assert 4.0 < multi < 6.0


def test_more_looks_means_a_lower_threshold():
    from core.sar.cfar import gamma_threshold

    thresholds = [gamma_threshold(looks, 1e-6) for looks in (1.0, 4.4, 16.0)]

    assert thresholds == sorted(thresholds, reverse=True)


# --------------------------------------------------------------------------
# Clutter estimation
# --------------------------------------------------------------------------


def test_clutter_mean_recovers_a_uniform_background():
    image = np.full((64, 64), 100.0)

    background = clutter_mean(image, CfarConfig(guard_half=2, train_half=5))

    assert np.allclose(background, 100.0)


def test_clutter_estimate_excludes_the_guard_band():
    """A target must not be allowed to raise its own threshold.

    This is the detail that makes CFAR work: without the guard band a bright
    ship contaminates the background estimate under itself and hides.
    """
    image = np.full((64, 64), 100.0)
    image[30:34, 30:34] = 100_000.0  # a very bright target

    background = clutter_mean(image, CfarConfig(guard_half=6, train_half=12))

    # At the target's centre the background must still read as sea.
    assert background[32, 32] == pytest.approx(100.0, rel=0.05)


def test_clutter_estimate_is_sane_at_the_scene_edge():
    """Zero-padding would fabricate darkness and ring the edge with detections."""
    image = np.full((64, 64), 100.0)

    background = clutter_mean(image, CfarConfig(guard_half=2, train_half=8))

    assert background[0, 0] == pytest.approx(100.0, rel=0.05)
    assert background[-1, -1] == pytest.approx(100.0, rel=0.05)


# --------------------------------------------------------------------------
# False alarm rate: the property that justifies the algorithm
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pfa", [1e-3, 1e-4, 1e-5])
def test_measured_false_alarm_rate_tracks_the_requested_pfa(pfa):
    """Empty sea, no targets: the measured rate should approximate the dial.

    Tolerance is an order of magnitude either way. The exponential clutter
    model is an idealisation of real speckle, so exactness is not expected -
    but the rate must move with the dial and stay in the right ballpark.
    """
    scene = make_scene((600, 600), seed=7)
    config = CfarConfig(guard_half=4, train_half=10, pfa=pfa)

    detections = cfar_detect(scene.amplitude, config)
    measured = measured_false_alarm_rate(detections)

    assert measured < pfa * 30, f"far too many false alarms: {measured:.2e}"
    assert measured < 0.01


def test_tightening_pfa_reduces_false_alarms():
    scene = make_scene((500, 500), seed=11)

    loose = cfar_detect(scene.amplitude, CfarConfig(guard_half=4, train_half=10, pfa=1e-3))
    strict = cfar_detect(scene.amplitude, CfarConfig(guard_half=4, train_half=10, pfa=1e-7))

    assert strict.sum() < loose.sum()


def test_empty_sea_produces_almost_no_detections():
    scene = make_scene((400, 400), seed=3)

    detections = cfar_detect(scene.amplitude, CfarConfig(pfa=1e-6))

    assert detections.mean() < 0.001


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_vessels_are_detected():
    targets = ship_grid((600, 600), rows=3, cols=3, length_px=20, width_px=6)
    scene = make_scene((600, 600), targets=targets, seed=5)

    detections = cfar_detect(scene.amplitude, CfarConfig())

    found = sum(
        1 for t in scene.targets if detections[t.row - 2 : t.row + 3, t.col - 6 : t.col + 7].any()
    )
    assert found == len(scene.targets)


def test_detections_land_on_the_targets():
    targets = ship_grid((500, 500), rows=2, cols=2, length_px=24, width_px=8)
    scene = make_scene((500, 500), targets=targets, seed=9)

    detections = cfar_detect(scene.amplitude, CfarConfig())

    # The overwhelming majority of detected pixels should be real targets.
    assert (detections & scene.truth).sum() / max(detections.sum(), 1) > 0.9


def test_a_faint_small_vessel_is_harder_than_a_bright_large_one():
    """Bounds the honest limitation: small low-RCS craft are the failure mode.

    Wooden dhows are abundant in the Gulf and are exactly this case.
    """
    bright = Target(250, 250, length_px=30, width_px=10, amplitude=1400.0)
    faint = Target(250, 250, length_px=3, width_px=2, amplitude=200.0)
    config = CfarConfig()

    bright_hits = cfar_detect(
        make_scene((500, 500), targets=[bright], seed=2).amplitude, config
    ).sum()
    faint_hits = cfar_detect(
        make_scene((500, 500), targets=[faint], seed=2).amplitude, config
    ).sum()

    assert bright_hits > faint_hits


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


def test_land_triggers_detections_when_unmasked():
    """Establishes why the mask is not optional."""
    scene = make_scene((400, 400), land_rows=120, seed=4)

    detections = cfar_detect(scene.amplitude, CfarConfig(guard_half=4, train_half=10))

    assert detections[:120, :].sum() > 0


def test_masking_land_removes_those_detections():
    scene = make_scene((400, 400), land_rows=120, seed=4)

    detections = cfar_detect(
        scene.amplitude, CfarConfig(guard_half=4, train_half=10), mask=scene.land
    )

    assert detections[:120, :].sum() == 0


def test_masked_land_does_not_blind_the_adjacent_sea():
    """Bright land inside a training ring would raise the coastal threshold.

    Left uncorrected, that hides every vessel along the shore - the exact
    stretch of water that matters most.
    """
    coast_row = 120
    ship = Target(coast_row + 25, 200, length_px=24, width_px=8)
    scene = make_scene((400, 400), targets=[ship], land_rows=coast_row, seed=6)
    config = CfarConfig()

    with_mask = cfar_detect(scene.amplitude, config, mask=scene.land)

    near_shore = with_mask[ship.row - 4 : ship.row + 5, ship.col - 12 : ship.col + 13]
    assert near_shore.any(), "a vessel just offshore was missed"


def test_mask_shape_is_validated():
    scene = make_scene((100, 100), seed=1)

    with pytest.raises(ValueError, match="does not match"):
        cfar_detect(scene.amplitude, mask=np.zeros((50, 50), dtype=bool))


# --------------------------------------------------------------------------
# Speckle model
# --------------------------------------------------------------------------


def test_speckle_preserves_the_mean_intensity():
    rng = np.random.default_rng(0)
    field = np.full((300, 300), 6400.0)

    speckled = speckle(field, IW_GRDH_LOOKS, rng)

    assert speckled.mean() == pytest.approx(6400.0, rel=0.02)


def test_speckle_is_multiplicative_not_additive():
    """Variance must scale with the signal, as real SAR speckle does."""
    rng = np.random.default_rng(0)

    dim = speckle(np.full((300, 300), 100.0), IW_GRDH_LOOKS, rng)
    bright = speckle(np.full((300, 300), 10_000.0), IW_GRDH_LOOKS, rng)

    assert bright.std() > dim.std() * 50


def test_synthetic_sea_has_the_expected_amplitude():
    scene = make_scene((300, 300), seed=1)

    assert scene.amplitude.mean() == pytest.approx(SEA_AMPLITUDE, rel=0.15)


def test_truth_marks_only_the_targets():
    targets = ship_grid((400, 400), rows=2, cols=2, length_px=20, width_px=6)
    scene = make_scene((400, 400), targets=targets, seed=1)

    assert scene.truth.any()
    assert scene.truth.mean() < 0.02
