"""Tests for decoding point detections from heatmaps, and matching them to truth."""

from __future__ import annotations

import numpy as np
import pytest

from core.inference.decode import decode_peaks, match_points
from core.labeling.targets import encode_targets


def test_decode_recovers_the_encoded_position():
    """Encoding then decoding must return the chip coordinate we started from."""
    record = {
        "supervision": "positive",
        "row": 1000.3,
        "col": 2000.0,
        "objects": [[20.0, -35.0, "vessel"]],
    }
    targets = encode_targets(record)

    peaks = decode_peaks(targets["heatmap"], targets["offset"], threshold=0.5)

    found = sorted((round(r, 3), round(c, 3)) for r, c, _ in peaks)
    assert found == [(64.3, 64.0), (84.3, 29.0)]


def test_decode_keeps_one_peak_per_blob_and_drops_weak_ones():
    heat = np.zeros((64, 64), dtype=np.float32)
    heat[10, 10], heat[10, 11], heat[11, 10] = 0.9, 0.8, 0.7  # one blob
    heat[40, 40] = 0.2  # below threshold
    offset = np.zeros((2, 64, 64), dtype=np.float32)

    peaks = decode_peaks(heat, offset, threshold=0.3)

    assert len(peaks) == 1
    row, col, score = peaks[0]
    assert (row, col) == (20.5, 20.5)
    assert score == pytest.approx(0.9)


def test_decode_orders_peaks_by_score():
    heat = np.zeros((64, 64), dtype=np.float32)
    heat[5, 5], heat[30, 30] = 0.4, 0.8

    peaks = decode_peaks(heat, np.zeros((2, 64, 64), dtype=np.float32), threshold=0.3)

    assert [p[2] for p in peaks] == pytest.approx([0.8, 0.4])


def test_match_points_pairs_within_tolerance_greedily_by_score():
    predictions = [(10.0, 10.0, 0.9), (50.0, 50.0, 0.8), (11.0, 10.0, 0.7)]
    truth = [(10.0, 11.0), (90.0, 90.0)]

    tp, fp, fn = match_points(predictions, truth, tolerance=5.0)

    assert (tp, fp, fn) == (1, 2, 1)


def test_match_points_can_leave_predictions_in_ignored_ground_uncounted():
    predictions = [(10.0, 10.0, 0.9), (40.0, 40.0, 0.8)]
    truth = [(10.0, 10.0)]

    tp, fp, fn = match_points(predictions, truth, tolerance=5.0, counts=lambda r, _c: r < 30)

    assert (tp, fp, fn) == (1, 0, 0)
