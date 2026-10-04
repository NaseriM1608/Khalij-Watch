"""Tests for the ONNX point-detector runner used by the product."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from core.inference.decode import decode_peaks
from core.inference.runner import PointDetection, _starts, detections_from_peaks, run_tiled
from core.labeling.targets import output_position


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


@pytest.mark.parametrize(
    ("tile", "overlap"),
    [(128, 31), (128, 33), (127, 31), (129, 33), (128, 128), (128, 130), (128, -2)],
)
def test_run_tiled_refuses_tilings_that_misplace_tiles(tile, overlap):
    image = np.zeros((2, 300, 300), dtype=np.float32)

    with pytest.raises(ValueError):
        run_tiled(_never_called, image, tile=tile, overlap=overlap)


@pytest.mark.parametrize(("tile", "overlap"), [(128, 32), (256, 64), (128, 64), (256, 32)])
def test_run_tiled_accepts_even_tilings(tile, overlap):
    heat, offset = run_tiled(
        _silent, np.zeros((2, 300, 300), dtype=np.float32), tile=tile, overlap=overlap
    )

    assert heat.shape == (150, 150)
    assert offset.shape == (2, 150, 150)


def test_runner_refuses_an_odd_overlap(monkeypatch):
    from core.inference import runner

    monkeypatch.setattr(runner.PointDetectorRunner, "__init__", _bare_init)
    detector = runner.PointDetectorRunner(None, overlap=31)
    detector.predict = _silent

    with pytest.raises(ValueError):
        detector.detect(np.zeros((2, 300, 300), dtype=np.float32), np.zeros((300, 300), bool))


def test_defaults_are_the_ones_chosen_against_gfw_for_kaggle_v2():
    from core.inference import runner

    assert (runner.DEFAULT_TILE, runner.DEFAULT_OVERLAP) == (256, 64)
    assert runner.DEFAULT_THRESHOLD == 0.60


def test_runner_uses_the_defaults_when_not_given(monkeypatch):
    import sys
    from types import SimpleNamespace

    from core.inference import runner

    fake_ort = SimpleNamespace(InferenceSession=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    detector = runner.PointDetectorRunner(None)

    assert (detector.tile, detector.overlap, detector.threshold) == (256, 64, 0.60)


def _bare_init(self, _model_path, *, threshold=0.45, tile=128, overlap=32):
    self.threshold, self.tile, self.overlap = threshold, tile, overlap


def _never_called(_batch):
    raise AssertionError("predict should not run")


def _silent(batch):
    h, w = batch.shape[2] // 2, batch.shape[3] // 2
    return np.full((1, 1, h, w), -10.0, dtype=np.float32), np.zeros((1, 2, h, w), np.float32)


def _encode_bright_pixels(batch):
    """A fake model: each bright input pixel encoded as core.labeling.targets encodes a vessel."""
    h, w = batch.shape[2] // 2, batch.shape[3] // 2
    logits = np.full((1, 1, h, w), -10.0, dtype=np.float32)
    offset = np.zeros((1, 2, h, w), dtype=np.float32)
    for row, col in zip(*np.nonzero(batch[0, 0] > 0.5), strict=True):
        cell_row, off_row = output_position(float(row))
        cell_col, off_col = output_position(float(col))
        logits[0, 0, cell_row, cell_col] = 10.0
        offset[0, :, cell_row, cell_col] = (off_row, off_col)
    return logits, offset


def _spaced(size, start):
    """Positions of both parities, at least 4 px apart, reaching both edges."""
    picked = []
    for p in [0, *range(start, size, 5), size - 1]:
        if not picked or p - picked[-1] >= 4:
            picked.append(p)
    return picked


@pytest.mark.parametrize(
    "shape", [(64, 64), (100, 129), (128, 128), (257, 300), (95, 97), (1000, 517)]
)
@pytest.mark.parametrize(("tile", "overlap"), [(128, 32), (256, 64)])
def test_run_tiled_and_decode_recover_every_pixel_position(shape, tile, overlap):
    height, width = shape
    rows, cols = _spaced(height, 3), _spaced(width, 2)
    image = np.zeros((2, height, width), dtype=np.float32)
    image[0, np.ix_(rows, cols)[0], np.ix_(rows, cols)[1]] = 1.0

    heat, offset = run_tiled(_encode_bright_pixels, image, tile=tile, overlap=overlap)
    peaks = decode_peaks(heat, offset, threshold=0.5)

    found = np.array(sorted((r, c) for r, c, _ in peaks))
    expected = np.array(sorted((r, c) for r in rows for c in cols), dtype=float)
    assert found.shape == expected.shape
    np.testing.assert_allclose(found, expected, atol=1e-3)


def test_heat_and_offset_at_an_overlapped_cell_come_from_the_same_tile():
    calls = []

    def predict(batch):
        tile = len(calls)
        calls.append(tile)
        h, w = batch.shape[2] // 2, batch.shape[3] // 2
        cols = np.arange(w, dtype=np.float32)
        # The first tile is sure on its left, the second on its right.
        slope = -0.1 if tile == 0 else 0.1
        logits = np.broadcast_to(slope * (cols - w / 2), (1, 1, h, w)).astype(np.float32)
        offset = np.full((1, 2, h, w), float(tile), dtype=np.float32)
        return logits, offset

    heat, offset = run_tiled(
        predict, np.zeros((2, 128, 224), dtype=np.float32), tile=128, overlap=32
    )

    assert len(calls) == 2  # tiles start at columns 0 and 96, overlapping in 96..127
    overlap_cells = slice(48, 64)
    winners = offset[0, :, overlap_cells]
    assert set(np.unique(winners)) == {0.0, 1.0}  # each tile wins somewhere
    cells = np.arange(heat.shape[1], dtype=np.float32)
    local = np.where(offset[0] == 0, cells, cells - 48)  # cell within the winning tile
    slope = np.where(offset[0] == 0, -0.1, 0.1)
    expected = 1.0 / (1.0 + np.exp(-(slope * (local - 32)).astype(np.float32)))
    np.testing.assert_allclose(heat, expected, atol=1e-6)
    np.testing.assert_array_equal(offset[1], offset[0])


@pytest.mark.parametrize(("tile", "overlap"), [(128, 32), (256, 64), (128, 64), (256, 32)])
def test_tile_starts_are_even_increasing_and_reach_the_end(tile, overlap):
    for size in range(1, 700):
        starts = _starts(size, tile, tile - overlap)

        if size <= tile:
            assert starts == [0]
            continue
        assert starts[0] == 0
        assert all(s % 2 == 0 for s in starts)
        assert all(b > a for a, b in pairwise(starts))
        assert all(b - a <= tile - overlap for a, b in pairwise(starts))
        assert starts[-1] + tile >= size
