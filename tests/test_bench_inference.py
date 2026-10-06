"""Tests for the CPU benchmark of the product model (Phase 8, efficiency)."""

from __future__ import annotations

import numpy as np

from scripts import bench_inference as bench


def test_the_synthetic_window_is_two_bands_of_sea_clutter_and_repeatable():
    first = bench.synthetic_window(64, seed=1)
    again = bench.synthetic_window(64, seed=1)
    assert first.shape == (2, 64, 64)
    assert first.dtype == np.float32
    assert np.array_equal(first, again)
    vv, vh = (float(np.median(band)) for band in first)
    assert 30 < vv < 120 and 10 < vh < 60  # sea levels measured on S1 (DN)
    assert first.max() > 10 * vv  # a few bright, ship-like targets


def test_peak_memory_is_reported_in_megabytes():
    assert bench.peak_memory_mb() > 10
