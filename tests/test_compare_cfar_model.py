"""Tests for scoring detectors against GFW cell counts, and tiled model inference."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.inference.runner import run_tiled
from scripts.compare_cfar_model import cell_key, match_cells, score_cells, summarise


def write_pass(folder, name, *, refs, cfar=(), model=(), area="fujairah"):
    record = {"area": area, "scene": name, "cfar": list(cfar), "model": list(model), "refs": refs}
    (folder / f"{area}__{name}.json").write_text(json.dumps(record), encoding="utf-8")


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


def test_match_cells_reports_reference_hits_per_cell():
    refs = [(56.50, 25.20, 2), (56.52, 25.20, 1)]
    preds = [(56.50, 25.20), (56.51, 25.20), (56.60, 25.30)]

    hits, fp = match_cells(preds, refs, neighbours=True)

    assert hits[cell_key(56.50, 25.20)] + hits[cell_key(56.52, 25.20)] == 2
    assert hits[cell_key(56.50, 25.20)] >= 1
    assert fp == 1


def test_one_prediction_is_not_a_hit_in_both_subsets(tmp_path):
    refs = [[56.50, 25.20, 1, True], [56.50, 25.20, 1, False]]
    write_pass(tmp_path, "a", refs=refs, cfar=[[56.50, 25.20]])

    cfar = summarise(tmp_path)["table"]["same_cell"]["all"]["cfar"]

    assert cfar["tp"] == 1
    assert cfar["tp_ais_matched"] + cfar["tp_unmatched"] == pytest.approx(1.0)
    assert cfar["recall_ais_matched"] == pytest.approx(0.5)
    assert cfar["recall_unmatched"] == pytest.approx(0.5)
    assert cfar["recall_unmatched_range"] == pytest.approx([0.0, 1.0])


def test_neighbour_rule_does_not_reuse_a_prediction_for_the_unmatched_subset(tmp_path):
    refs = [[56.50, 25.20, 1, True], [56.51, 25.20, 1, False]]
    write_pass(tmp_path, "a", refs=refs, cfar=[[56.50, 25.20]])

    cfar = summarise(tmp_path)["table"]["neighbour_cells"]["all"]["cfar"]

    assert cfar["tp"] == 1
    assert cfar["recall_ais_matched"] == pytest.approx(1.0)
    assert cfar["recall_unmatched"] == pytest.approx(0.0)
    assert cfar["recall_unmatched_range"] == pytest.approx([0.0, 0.0])


def test_subset_hits_add_up_to_the_joint_true_positives(tmp_path):
    refs = [
        [56.50, 25.20, 2, True],
        [56.50, 25.20, 1, False],
        [56.51, 25.20, 1, False],
        [56.60, 25.30, 1, True],
    ]
    write_pass(tmp_path, "a", refs=refs, cfar=[[56.50, 25.20], [56.50, 25.20], [56.51, 25.21]])

    for rule in ("same_cell", "neighbour_cells"):
        cfar = summarise(tmp_path)["table"][rule]["all"]["cfar"]
        assert cfar["tp_ais_matched"] + cfar["tp_unmatched"] == pytest.approx(cfar["tp"])
        assert cfar["ref_matched"] == 3
        assert cfar["ref_unmatched"] == 2


def test_summarise_thresholds_model_scores_but_not_cfar(tmp_path):
    refs = [[56.50, 25.20, 1, True], [56.60, 25.30, 1, True]]
    model = [[56.50, 25.20, 0.9], [56.60, 25.30, 0.2]]
    cfar = [[56.50, 25.20], [56.60, 25.30], [56.70, 25.40]]
    write_pass(tmp_path, "a", refs=refs, cfar=cfar, model=model)

    table = summarise(tmp_path, threshold=0.45)["table"]["same_cell"]["all"]

    assert (table["model"]["tp"], table["model"]["fp"], table["model"]["fn"]) == (1, 0, 1)
    assert (table["cfar"]["tp"], table["cfar"]["fp"], table["cfar"]["fn"]) == (2, 1, 0)
    assert table["model"]["precision"] == pytest.approx(1.0)
    assert table["model"]["recall"] == pytest.approx(0.5)
    assert table["model"]["f1"] == pytest.approx(2 / 3)


def test_summarise_reports_each_area_and_ignores_its_own_output(tmp_path):
    write_pass(tmp_path, "a", refs=[[56.50, 25.20, 1, True]], cfar=[[56.50, 25.20]])
    write_pass(tmp_path, "b", refs=[[55.30, 25.50, 1, False]], area="sharjah_dubai")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    summary = summarise(tmp_path)

    assert summary["scenes"] == 2
    rows = summary["table"]["same_cell"]
    assert set(rows) == {"all", "fujairah", "sharjah_dubai"}
    assert rows["fujairah"]["cfar"]["recall"] == pytest.approx(1.0)
    assert rows["sharjah_dubai"]["cfar"]["recall"] == pytest.approx(0.0)
    assert rows["all"]["cfar"]["tp"] == 1


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
