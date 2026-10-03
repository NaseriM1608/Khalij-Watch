"""Tests for scoring detectors against GFW cell counts, and tiled model inference."""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from core.db.spatial import BBox
from core.inference.runner import run_tiled
from core.labeling.gfw import GfwCell
from scripts import compare_cfar_model as compare
from scripts.compare_cfar_model import (
    cell_key,
    gfw_days,
    match_cells,
    reference_cells,
    score_cells,
    summarise,
    unmasked,
)


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


def test_summary_of_a_missing_folder_fails_cleanly_without_creating_it(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["compare", "--summary", "--out", str(out)])

    assert compare.main() == 1
    assert not out.exists()
    assert "no results" in capsys.readouterr().out


def test_summary_of_an_empty_folder_fails_without_writing_a_summary(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare", "--summary", "--out", str(tmp_path)])

    assert compare.main() == 1
    assert (tmp_path / "summary.json").read_text(encoding="utf-8") == "{}"


def test_gfw_days_cover_a_pass_well_inside_one_day():
    acquired = dt.datetime(2024, 5, 16, 2, 12, tzinfo=dt.UTC)

    assert gfw_days(acquired) == (dt.date(2024, 5, 16), dt.date(2024, 5, 17))


def test_gfw_days_reach_into_the_next_day_for_a_pass_just_before_midnight():
    acquired = dt.datetime(2024, 5, 16, 23, 59, 30, tzinfo=dt.UTC)

    assert gfw_days(acquired) == (dt.date(2024, 5, 16), dt.date(2024, 5, 18))


def test_gfw_days_reach_back_a_day_for_a_pass_just_after_midnight():
    acquired = dt.datetime(2024, 5, 17, 0, 1, tzinfo=dt.UTC)

    assert gfw_days(acquired) == (dt.date(2024, 5, 16), dt.date(2024, 5, 18))


def test_unmasked_keeps_only_in_window_unmasked_pixels():
    masked = np.zeros((4, 5), dtype=bool)
    masked[2, 3] = True

    assert unmasked(masked, 1.2, 1.4)
    assert unmasked(masked, -0.4, 0.0)  # rounds to pixel (0, 0)
    assert not unmasked(masked, 2.4, 2.6)  # rounds to the masked pixel (2, 3)
    assert not unmasked(masked, -0.6, 1.0)
    assert not unmasked(masked, 1.0, 4.6)
    assert not unmasked(masked, 3.6, 1.0)


class FakeScene:
    """North-up raster: pixel corners every 0.001 degree from (56.0, 26.0)."""

    def raster_coords(self, lons, lats):
        return (
            np.array([(26.0 - lat) / 0.001 for lat in lats]),
            np.array([(lon - 56.0) / 0.001 for lon in lons]),
        )


def gfw_cell(lon, lat, detections=1, matched=True):
    hour = dt.datetime(2024, 5, 16, 2, tzinfo=dt.UTC)
    return GfwCell(lat=lat, lon=lon, hour=hour, detections=detections, matched=matched)


def test_reference_cells_place_a_cell_centre_on_the_pixel_it_falls_in():
    box = BBox(55.9, 25.9, 56.1, 26.1)
    # Corner-based (12.5, 7.5) is the centre of scene pixel (12, 7); the window starts at (10, 5).
    cell = gfw_cell(56.0075, 25.9875, detections=2, matched=False)
    masked = np.ones((6, 6), dtype=bool)
    masked[2, 2] = False

    refs, dropped = reference_cells([cell], box, FakeScene(), (10, 5), masked)

    assert refs == [(56.0075, 25.9875, 2, False)]
    assert dropped == 0


def test_reference_cells_drop_and_count_masked_or_off_window_cells():
    box = BBox(55.9, 25.9, 56.1, 26.1)
    masked = np.zeros((6, 6), dtype=bool)
    masked[2, 2] = True
    on_mask = gfw_cell(56.0075, 25.9875, detections=2)
    off_window = gfw_cell(56.0995, 25.9875, detections=3)
    kept = gfw_cell(56.0065, 25.9875)

    refs, dropped = reference_cells([on_mask, off_window, kept], box, FakeScene(), (10, 5), masked)

    assert refs == [(56.0065, 25.9875, 1, True)]
    assert dropped == 5


def test_reference_cells_skip_cells_outside_the_box_without_counting_them():
    box = BBox(56.0, 25.98, 56.01, 26.0)
    masked = np.zeros((6, 6), dtype=bool)

    refs, dropped = reference_cells(
        [gfw_cell(56.2, 25.99, detections=4)], box, FakeScene(), (10, 5), masked
    )

    assert refs == []
    assert dropped == 0


def scenes(count):
    return [f"S1A_IW_GRDH_1SDV_2024{i:04d}T020000.SAFE" for i in range(count)]


def test_spread_takes_per_area_passes_from_each_area_with_more():
    pairs = [("a", s) for s in scenes(21)] + [("b", s) for s in scenes(5)]

    chosen = compare.spread(pairs, 3)

    assert [a for a, _ in chosen] == ["a"] * 3 + ["b"] * 3


def test_spread_keeps_every_pass_when_an_area_has_no_more_than_per_area():
    pairs = [("a", s) for s in scenes(4)]

    assert compare.spread(pairs, 4) == sorted(pairs)
    assert compare.spread(pairs, 8) == sorted(pairs)


def test_spread_is_deterministic_whatever_the_input_order():
    pairs = [("a", s) for s in scenes(21)]

    assert compare.spread(pairs, 8) == compare.spread(list(reversed(pairs)), 8)


@pytest.mark.parametrize(
    ("count", "indices"),
    [(21, [0, 2, 5, 7, 10, 13, 15, 18]), (16, [0, 2, 4, 6, 8, 10, 12, 14])],
)
def test_spread_picks_the_same_scene_indices_as_run_1(count, indices):
    names = scenes(count)

    chosen = compare.spread([("fujairah", s) for s in names], 8)

    assert chosen == [("fujairah", names[i]) for i in indices]


def test_test_pairs_keeps_only_test_records_deduplicated_and_sorted(tmp_path):
    records = [
        {"split": "test", "area": "b", "scene": "S2"},
        {"split": "test", "area": "a", "scene": "S9"},
        {"split": "test", "area": "b", "scene": "S2"},
        {"split": "train", "area": "a", "scene": "test"},
        {"split": "val", "area": "a", "scene": "S1", "note": "test"},
    ]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    assert compare.test_pairs(index) == [("a", "S9"), ("b", "S2")]


def test_scoring_with_no_predictions_misses_every_reference():
    assert score_cells([], [(56.50, 25.20, 2)], neighbours=True) == (0, 0, 2)


def test_scoring_with_no_references_makes_every_prediction_a_false_alarm():
    assert score_cells([(56.50, 25.20), (56.60, 25.30)], [], neighbours=True) == (0, 2, 0)


def test_a_reference_with_count_zero_takes_no_prediction():
    assert score_cells([(56.50, 25.20)], [(56.50, 25.20, 0)], neighbours=True) == (0, 1, 0)


def test_cells_west_and_south_of_zero_match_and_take_neighbours():
    refs = [(-0.02, -0.03, 1), (-0.05, -0.05, 1)]
    preds = [(-0.021, -0.029), (-0.04, -0.05)]

    assert cell_key(-0.021, -0.029) == (-3, -2)
    assert score_cells(preds, refs, neighbours=False) == (1, 1, 1)
    assert score_cells(preds, refs, neighbours=True) == (2, 0, 0)


def test_duplicate_reference_rows_for_one_cell_are_summed():
    refs = [(56.50, 25.20, 1), (56.50, 25.20, 2)]
    preds = [(56.50, 25.20)] * 4

    hits, fp = match_cells(preds, refs, neighbours=False)

    assert hits[cell_key(56.50, 25.20)] == 3
    assert fp == 1


@pytest.mark.parametrize("neighbours", [False, True])
def test_true_and_false_positives_account_for_every_prediction(neighbours):
    refs = [(56.50, 25.20, 2), (56.52, 25.20, 1), (56.70, 25.40, 3)]
    preds = [(56.50, 25.20), (56.51, 25.20), (56.51, 25.21), (56.60, 25.30), (56.70, 25.40)]

    tp, fp, _ = score_cells(preds, refs, neighbours=neighbours)

    assert tp + fp == len(preds)


def test_split_hits_of_no_references_is_all_zero():
    assert compare.split_hits(Counter(), []) == {
        "matched": 0.0,
        "unmatched": 0.0,
        "unmatched_min": 0,
        "unmatched_max": 0,
    }


def test_split_hits_in_a_matched_only_cell_are_all_matched():
    key = cell_key(56.50, 25.20)

    split = compare.split_hits(Counter({key: 2}), [(56.50, 25.20, 3, True)])

    assert split == {"matched": 2.0, "unmatched": 0.0, "unmatched_min": 0, "unmatched_max": 0}


def test_split_hits_in_an_unmatched_only_cell_are_all_unmatched():
    key = cell_key(56.50, 25.20)

    split = compare.split_hits(Counter({key: 2}), [(56.50, 25.20, 3, False)])

    assert split == {"matched": 0.0, "unmatched": 2.0, "unmatched_min": 2, "unmatched_max": 2}


def test_summarise_of_no_results_reports_zero_scenes(tmp_path):
    summary = summarise(tmp_path)

    assert summary["scenes"] == 0
    assert summary["table"]["same_cell"]["all"]["model"]["f1"] == 0.0


def test_summarise_counts_a_model_score_exactly_at_the_threshold(tmp_path):
    write_pass(tmp_path, "a", refs=[[56.50, 25.20, 1, True]], model=[[56.50, 25.20, 0.45]])

    model = summarise(tmp_path, threshold=0.45)["table"]["same_cell"]["all"]["model"]

    assert (model["tp"], model["fp"], model["fn"]) == (1, 0, 0)


def test_main_skips_passes_already_compared_and_writes_new_ones(tmp_path, monkeypatch):
    import core.config

    done, new = (
        "S1A_IW_GRDH_1SDV_20240516T021200_A.SAFE",
        "S1A_IW_GRDH_1SDV_20240528T021200_B_COG.SAFE",
    )
    index = tmp_path / "index.jsonl"
    index.write_text(
        "".join(
            json.dumps({"split": "test", "area": "fujairah", "scene": s}) + "\n"
            for s in (done, new)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    write_pass(out, done.removesuffix(".SAFE"), refs=[[56.50, 25.20, 1, True]])
    compared = []

    def fake_compare_scene(area, scene_name, session, client):
        compared.append((area, scene_name, session, client))
        return {
            "area": area,
            "scene": scene_name,
            "cfar": [[56.50, 25.20]],
            "model": [],
            "refs": [[56.50, 25.20, 1, True]],
            "seconds": 0.1,
        }

    monkeypatch.setattr(compare, "compare_scene", fake_compare_scene)
    monkeypatch.setitem(sys.modules, "gfwapiclient", SimpleNamespace(Client=lambda **_: "client"))
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=lambda *_, **__: "session"),
    )
    monkeypatch.setattr(core.config, "load_settings", lambda: None)
    monkeypatch.setenv("GFW_API_TOKEN", "not-a-token")
    monkeypatch.setattr(sys, "argv", ["compare", "--index", str(index), "--out", str(out)])

    assert compare.main() == 0
    assert compared == [("fujairah", new, "session", "client")]
    assert (out / "fujairah__S1A_IW_GRDH_1SDV_20240528T021200_B_COG.json").exists()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["scenes"] == 2
