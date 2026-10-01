"""Tests for scoring detectors against GFW cell counts, and tiled model inference."""

from __future__ import annotations

import numpy as np

from core.inference.runner import run_tiled
from scripts.compare_cfar_model import cell_key, score_cells


def test_same_cell_detections_match_up_to_the_reference_count():
    refs = [(56.50, 25.20, 2)]
    preds = [(56.501, 25.201), (56.499, 25.199), (56.502, 25.202)]

    assert score_cells(preds, refs, neighbours=False) == (2, 1, 0)


def test_neighbour_matching_takes_up_a_detection_one_cell_over():
    refs = [(56.50, 25.20, 1)]
    preds = [(56.51, 25.20)]

    assert score_cells(preds, refs, neighbours=False) == (0, 1, 1)
    assert score_cells(preds, refs, neighbours=True) == (1, 0, 0)


def test_exact_matches_are_used_before_neighbours():
    refs = [(56.50, 25.20, 1), (56.51, 25.20, 1)]
    preds = [(56.50, 25.20), (56.51, 25.20)]

    assert score_cells(preds, refs, neighbours=True) == (2, 0, 0)


def test_cell_key_matches_gfw_cell_centres():
    assert cell_key(56.504, 25.196) == cell_key(56.50, 25.20)
    assert cell_key(56.506, 25.20) != cell_key(56.50, 25.20)


def test_run_tiled_stitches_tiles_back_to_one_half_resolution_map():
    seen = []

    def predict(batch):
        seen.append(batch.shape)
        h, w = batch.shape[2] // 2, batch.shape[3] // 2
        logits = np.full((1, 1, h, w), -10.0, dtype=np.float32)
        logits[0, 0, :, :] = batch[0, 0, ::2, ::2] * 20 - 10  # bright input -> high score
        return logits, np.zeros((1, 2, h, w), dtype=np.float32)

    image = np.zeros((2, 300, 260), dtype=np.float32)
    image[0, 250, 200] = 1.0  # one bright target near a tile boundary

    heat, offset = run_tiled(predict, image, tile=128, overlap=32)

    assert heat.shape == (150, 130)
    assert offset.shape == (2, 150, 130)
    assert np.unravel_index(heat.argmax(), heat.shape) == (125, 100)
    assert heat.max() > 0.99
    assert all(s[2] % 32 == 0 and s[3] % 32 == 0 for s in seen)
