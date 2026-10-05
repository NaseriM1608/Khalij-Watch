"""The dark rate and its interval from resampling whole passes.

The dark rate is the share of kept detections (after the pre-filters) with no
AIS candidate in the gate, pooled over passes as a ratio of sums. Its 95%
interval comes from resampling whole passes, as in the detector comparison
(scripts/compare_cfar_model.py: 10,000 resamples, seed 0), because detections
within one pass share its weather, traffic and AIS reception (PLAN §17,
2026-10-04 "Phase 6 feasibility").
"""

from __future__ import annotations

import pytest

from core.fusion.rate import dark_rate


def test_the_rate_is_a_ratio_of_sums_over_passes():
    estimate = dark_rate(dark=[1, 3], kept=[10, 10], resamples=200)
    assert estimate.rate == pytest.approx(0.2)
    assert (estimate.dark, estimate.kept, estimate.passes) == (4, 20, 2)


def test_the_interval_brackets_the_rate_and_stays_in_0_1():
    estimate = dark_rate(dark=[0, 2, 5, 1, 9], kept=[20, 30, 25, 10, 40], resamples=2000)
    assert 0.0 <= estimate.low <= estimate.rate <= estimate.high <= 1.0
    assert estimate.low < estimate.high


def test_the_interval_is_reproducible():
    a = dark_rate(dark=[0, 2, 5, 1, 9], kept=[20, 30, 25, 10, 40], resamples=500)
    b = dark_rate(dark=[0, 2, 5, 1, 9], kept=[20, 30, 25, 10, 40], resamples=500)
    assert (a.low, a.high) == (b.low, b.high)


def test_identical_passes_give_a_zero_width_interval():
    estimate = dark_rate(dark=[2, 2, 2], kept=[10, 10, 10], resamples=300)
    assert estimate.low == pytest.approx(0.2)
    assert estimate.high == pytest.approx(0.2)


def test_passes_without_kept_detections_do_not_break_the_resampling():
    estimate = dark_rate(dark=[0, 2], kept=[0, 10], resamples=300)
    assert estimate.rate == pytest.approx(0.2)
    assert estimate.low == pytest.approx(0.2) and estimate.high == pytest.approx(0.2)


def test_no_kept_detections_give_no_rate():
    estimate = dark_rate(dark=[0, 0], kept=[0, 0], resamples=100)
    assert (estimate.rate, estimate.low, estimate.high) == (None, None, None)


def test_inconsistent_counts_are_rejected():
    with pytest.raises(ValueError, match="length"):
        dark_rate(dark=[1], kept=[1, 2])
    with pytest.raises(ValueError, match="dark"):
        dark_rate(dark=[3], kept=[2])
    with pytest.raises(ValueError, match="negative"):
        dark_rate(dark=[-1], kept=[2])


def test_no_dark_detection_gives_an_exact_upper_bound_not_a_zero_width_interval():
    estimate = dark_rate(dark=[0, 0, 0], kept=[100, 200, 100], resamples=100)
    assert estimate.rate == 0.0 and estimate.low == 0.0
    assert estimate.high == pytest.approx(1 - 0.025 ** (1 / 400))  # one-sided 97.5% Clopper-Pearson


def test_all_dark_gives_an_exact_lower_bound():
    estimate = dark_rate(dark=[5, 5], kept=[5, 5], resamples=100)
    assert estimate.high == 1.0
    assert estimate.low == pytest.approx(0.025 ** (1 / 10))
