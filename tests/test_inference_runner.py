"""Tests for the ONNX point-detector runner used by the product."""

from __future__ import annotations

import numpy as np

from core.inference.runner import PointDetection, detections_from_peaks, run_tiled


def test_run_tiled_is_shared_with_the_comparison_script():
    from scripts import compare_cfar_model

    assert compare_cfar_model.run_tiled is run_tiled


def test_masked_and_out_of_window_peaks_are_dropped():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    peaks = [(10.0, 10.0, 0.9), (50.0, 50.0, 0.8), (150.0, 10.0, 0.7)]

    kept = detections_from_peaks(peaks, mask, threshold=0.5)

    assert kept == [(10.0, 10.0, 0.9)]


def test_peaks_below_threshold_are_dropped():
    mask = np.zeros((100, 100), dtype=bool)

    assert detections_from_peaks([(10.0, 10.0, 0.3)], mask, threshold=0.45) == []


def test_point_detection_satisfies_the_storage_protocol():
    point = PointDetection(lon=56.5, lat=25.2, confidence=0.8)

    assert (point.length_m, point.width_m, point.heading_deg) == (None, None, None)
    assert (point.pixel_count, point.peak_amplitude) == (None, None)
