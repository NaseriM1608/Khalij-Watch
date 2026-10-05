"""Tests for scoring detectors against GFW cell counts, and tiled model inference."""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
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


def test_split_pairs_keeps_only_that_split_deduplicated_and_sorted(tmp_path):
    records = [
        {"split": "test", "area": "b", "scene": "S2"},
        {"split": "test", "area": "a", "scene": "S9"},
        {"split": "test", "area": "b", "scene": "S2"},
        {"split": "train", "area": "a", "scene": "test"},
        {"split": "val", "area": "a", "scene": "S1", "note": "test"},
    ]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    assert compare.split_pairs(index, "test") == [("a", "S9"), ("b", "S2")]
    assert compare.split_pairs(index, "val") == [("a", "S1")]


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


class GfwError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def offline_main(tmp_path, monkeypatch):
    """main() with the index, model files, GFW, ONNX and compare_scene replaced by fakes."""
    import core.config

    scenes = [
        "S1A_IW_GRDH_1SDV_20240516T021200_A.SAFE",
        "S1A_IW_GRDH_1SDV_20240528T021200_B_COG.SAFE",
    ]
    index = tmp_path / "index.jsonl"
    index.write_text(
        "".join(
            json.dumps({"split": "val", "area": "fujairah", "scene": s}) + "\n" for s in scenes
        ),
        encoding="utf-8",
    )
    models = {}
    for name in ("a", "b"):
        models[name] = tmp_path / f"{name}.onnx"
        models[name].write_bytes(name.encode())
    state = SimpleNamespace(
        index=index,
        out=tmp_path / "out",
        scenes=scenes,
        models=models,
        compared=[],
        fail=set(),
        error=GfwError("unprocessable request", status_code=422),
        clients=[],
    )
    state.out.mkdir()

    def fake_compare_scene(area, scene_name, loaded, _client):
        if scene_name in state.fail:
            raise state.error
        state.compared.append((scene_name, [(spec.label, session) for spec, _, session in loaded]))
        return {
            "area": area,
            "scene": scene_name,
            "cfar": [[56.50, 25.20]],
            "models": {
                spec.label: {**prov, "peaks": [[56.50, 25.20, 0.9]], "seconds": 0.1}
                for spec, prov, _ in loaded
            },
            "refs": [[56.50, 25.20, 1, True]],
            "seconds": 0.1,
        }

    monkeypatch.setattr(compare, "compare_scene", fake_compare_scene)
    monkeypatch.setitem(
        sys.modules,
        "gfwapiclient",
        SimpleNamespace(Client=lambda **kwargs: state.clients.append(kwargs) or "client"),
    )
    monkeypatch.setitem(
        sys.modules, "gfwapiclient.exceptions", SimpleNamespace(GFWAPIClientError=GfwError)
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=lambda path, **__: f"session:{Path(path).name}"),
    )
    monkeypatch.setattr(core.config, "load_settings", lambda: None)
    monkeypatch.setenv("GFW_API_TOKEN", "not-a-token")

    def run(*extra):
        argv = ["compare", "--index", str(index), "--out", str(state.out), "--split", "val"]
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        return compare.main()

    state.run = run
    return state


def test_main_runs_every_model_on_each_pass_once(offline_main):
    m = offline_main

    assert m.run("--model", f"a={m.models['a']}", "--model", f"b={m.models['b']}@256/64") == 0

    assert [scene for scene, _ in m.compared] == m.scenes
    assert m.compared[0][1] == [("a", "session:a.onnx"), ("b", "session:b.onnx")]
    result = json.loads(
        (m.out / "fujairah__S1A_IW_GRDH_1SDV_20240528T021200_B_COG.json").read_text()
    )
    assert result["split"] == "val"
    assert (result["models"]["b"]["tile"], result["models"]["b"]["overlap"]) == (256, 64)
    summary = json.loads((m.out / "summary.json").read_text(encoding="utf-8"))
    assert summary["scenes"] == 2
    assert set(summary["table"]["same_cell"]["all"]) == {"cfar", "a", "b"}


def test_main_resumes_only_passes_compared_with_the_same_models(offline_main):
    m = offline_main
    assert m.run("--model", f"a={m.models['a']}") == 0
    m.compared.clear()

    assert m.run("--model", f"a={m.models['a']}") == 0
    assert m.compared == []

    m.models["a"].write_bytes(b"retrained weights")
    assert m.run("--model", f"a={m.models['a']}") == 0
    assert [scene for scene, _ in m.compared] == m.scenes


def test_main_treats_a_gfw_error_about_one_request_as_one_failed_pass(offline_main, capsys):
    m = offline_main
    m.fail.add(m.scenes[0])

    assert m.run("--model", f"a={m.models['a']}") == 0

    assert [scene for scene, _ in m.compared] == [m.scenes[1]]
    assert "1 passes failed" in capsys.readouterr().out


@pytest.mark.parametrize("status", [429, 524, None])
def test_main_stops_at_a_gfw_lock_or_timeout_instead_of_sending_the_next_report(
    offline_main, capsys, status
):
    """GFW computes one report at a time; after a 429, a 5xx or a timeout the next would meet the lock."""
    m = offline_main
    m.fail.add(m.scenes[0])
    m.error = GfwError("busy", status_code=status)

    m.run("--model", f"a={m.models['a']}")

    assert m.compared == []
    assert "stopped" in capsys.readouterr().out


def test_main_says_when_the_report_never_reached_gfw(offline_main, capsys):
    """A connection that never opened holds no GFW lock: the runner may retry soon."""
    import httpx

    m = offline_main
    m.fail.add(m.scenes[0])
    m.error = GfwError("Connection error.")
    m.error.__cause__ = httpx.ConnectError("name resolution failed")

    m.run("--model", f"a={m.models['a']}")

    out = capsys.readouterr().out
    assert m.compared == []
    assert "stopped: GFW unreachable" in out


def test_main_counts_an_unexpected_error_as_one_failed_pass(offline_main, capsys):
    """One pass that always fails must not keep every later pass from being compared."""
    m = offline_main
    m.fail.add(m.scenes[0])
    m.error = KeyError("lat")

    assert m.run("--model", f"a={m.models['a']}") == 0

    assert [scene for scene, _ in m.compared] == [m.scenes[1]]
    out = capsys.readouterr().out
    assert "KeyError" in out
    assert "1 passes failed" in out


def test_main_gives_each_pass_its_own_gfw_client_with_a_long_timeout(offline_main):
    """Each pass runs its own event loop; a client's pooled connection dies with the loop.

    The timeout is longer than GFW's own gateway (about 100 s, then 524).
    """
    m = offline_main

    m.run("--model", f"a={m.models['a']}")

    assert m.clients == [{"access_token": "not-a-token", "timeout": 180.0}] * len(m.scenes)


def test_main_treats_a_catalogue_outage_as_one_failed_pass(offline_main, capsys):
    from core.sar.catalogue import CatalogueError

    m = offline_main
    m.fail.add(m.scenes[0])
    m.error = CatalogueError("catalogue unreachable")

    assert m.run("--model", f"a={m.models['a']}") == 0

    assert [scene for scene, _ in m.compared] == [m.scenes[1]]
    assert "1 passes failed" in capsys.readouterr().out


def test_main_redoes_a_pass_whose_result_was_cut_short(offline_main):
    m = offline_main
    assert m.run("--model", f"a={m.models['a']}") == 0
    broken = m.out / "fujairah__S1A_IW_GRDH_1SDV_20240516T021200_A.json"
    broken.write_bytes(b"\0\0\0\0")  # what a power cut can leave behind
    m.compared.clear()

    assert m.run("--model", f"a={m.models['a']}") == 0

    assert [scene for scene, _ in m.compared] == [m.scenes[0]]
    assert compare.load_result(broken)["models"]["a"]["tile"] == 128


def test_summary_totals_each_models_seconds_over_passes(offline_main):
    m = offline_main

    assert m.run("--model", f"a={m.models['a']}") == 0

    summary = json.loads((m.out / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["a"]["seconds_total"] == pytest.approx(0.2)
    assert "seconds" not in summary["models"]["a"]


def test_summarise_refuses_a_threshold_for_an_unknown_model(tmp_path):
    write_result(tmp_path, "s1", models={"a": ("x", 128, [])}, refs=[[56.50, 25.20, 1, True]])

    with pytest.raises(ValueError, match="e_14"):
        summarise(tmp_path, thresholds={"e_14": 0.3})


def test_bootstrap_states_the_threshold_it_was_computed_at(tmp_path):
    write_result(tmp_path, "s1", models={"a": ("x", 128, [])}, refs=[[56.50, 25.20, 1, True]])

    summary = summarise(tmp_path, thresholds={"a": 0.3}, resamples=10)

    assert summary["bootstrap"]["a"]["same_cell"]["threshold"] == pytest.approx(0.3)


def test_main_compares_the_passes_listed_in_a_pairs_file(offline_main, tmp_path):
    m = offline_main
    listed = "S1D_IW_GRDH_1SDV_20260910T020650_C_COG.SAFE"
    pairs = tmp_path / "s1cd_2026-09.txt"
    pairs.write_text(f"# new satellites\n\nfujairah {listed}\n", encoding="utf-8")

    assert m.run("--model", f"a={m.models['a']}", "--pairs", str(pairs)) == 0

    assert [scene for scene, _ in m.compared] == [listed]
    result = json.loads((m.out / f"fujairah__{listed.removesuffix('.SAFE')}.json").read_text())
    assert result["split"] == "s1cd_2026-09"


def test_read_pairs_refuses_an_unknown_area(tmp_path):
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("atlantis S1C_X.SAFE\n", encoding="utf-8")

    with pytest.raises(ValueError, match="atlantis"):
        compare.read_pairs(pairs)


def test_read_pairs_accepts_the_iranian_validation_boxes(tmp_path):
    from core.areas import IRAN_AREAS, STUDY_AREAS

    pairs = tmp_path / "iran.txt"
    pairs.write_text("bandar_abbas S1A_X.SAFE\njask S1A_Y.SAFE  # open sea\n", encoding="utf-8")

    assert compare.read_pairs(pairs) == [("bandar_abbas", "S1A_X.SAFE"), ("jask", "S1A_Y.SAFE")]
    assert compare.AREAS == {**STUDY_AREAS, **IRAN_AREAS}


def test_summarise_leaves_out_passes_gfw_did_not_cover(tmp_path):
    refs = [[56.50, 25.20, 1, True]]
    write_result(tmp_path, "s1", models={"a": ("x", 128, [])}, refs=refs, cfar=[[56.50, 25.20]])
    write_result(tmp_path, "s2", models={"a": ("x", 128, [])}, refs=[], cfar=[[56.60, 25.30]])
    for name, seen in (("s1", 1), ("s2", 0)):
        path = tmp_path / f"fujairah__{name}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "gfw_in_box": seen}), encoding="utf-8")

    summary = summarise(tmp_path)

    assert summary["scenes"] == 1
    assert summary["excluded_no_gfw"] == ["s2"]


def test_summarise_records_what_each_left_out_pass_detected(tmp_path):
    """In a quiet box a pass GFW covered can be empty: its detections are then all false alarms."""
    peaks = [[56.60, 25.30, 0.9], [56.61, 25.31, 0.1]]
    write_result(tmp_path, "s1", models={"a": ("x", 128, [])}, refs=[[56.50, 25.20, 1, True]])
    write_result(
        tmp_path, "s2", models={"a": ("x", 128, peaks)}, refs=[], cfar=[[56.60, 25.30]], area="jask"
    )
    path = tmp_path / "jask__s2.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "gfw_in_box": 0}), encoding="utf-8")

    summary = summarise(tmp_path)

    assert summary["excluded_no_gfw_detections"] == [
        {"area": "jask", "scene": "s2", "cfar": 1, "models": {"a": 1}}
    ]
    assert summary["table"]["same_cell"]["all"]["cfar"]["fp"] == 0


def test_sea_levels_are_medians_of_valid_unmasked_pixels():
    bands = np.zeros((2, 4, 4), dtype=np.uint16)
    bands[0] = [[10, 20, 30, 40]] * 4
    bands[1] = 5
    bands[1, 0, 0] = 0  # nodata in one band drops the pixel from both
    masked = np.zeros((4, 4), dtype=bool)
    masked[:, 3] = True  # coast: the 40 DN column

    assert compare.sea_levels(bands, masked) == [20.0, 5.0]


def test_parse_model_spec_reads_label_path_and_tiling():
    spec = compare.parse_model_spec(r"e14=E:\runs\epoch_14.onnx@256/64")

    assert (spec.label, spec.path, spec.tile, spec.overlap) == (
        "e14",
        Path(r"E:\runs\epoch_14.onnx"),
        256,
        64,
    )
    plain = compare.parse_model_spec("models/x.onnx")
    assert (plain.label, plain.tile, plain.overlap) == ("model", 128, 32)


@pytest.mark.parametrize(
    "text", ["a=x.onnx@256", "a=x.onnx@256/x", "a=x.onnx@128/31", "cfar=x.onnx", "=x.onnx"]
)
def test_parse_model_spec_refuses_bad_specs(text):
    with pytest.raises(ValueError):
        compare.parse_model_spec(text)


def write_result(folder, scene, *, models, refs, cfar=(), area="fujairah"):
    """A result in the several-models format; ``models`` maps label -> (sha, tile, peaks)."""
    record = {
        "area": area,
        "scene": scene,
        "split": "val",
        "cfar": list(cfar),
        "models": {
            label: {"path": f"{label}.onnx", "sha256": sha, "tile": tile, "overlap": 32, "peaks": p}
            for label, (sha, tile, p) in models.items()
        },
        "refs": refs,
    }
    (folder / f"{area}__{scene}.json").write_text(json.dumps(record), encoding="utf-8")


def test_summarise_scores_each_model_at_its_own_threshold(tmp_path):
    refs = [[56.50, 25.20, 1, True], [56.60, 25.30, 1, False]]
    peaks = [[56.50, 25.20, 0.9], [56.60, 25.30, 0.6]]
    write_result(tmp_path, "s1", models={"a": ("x", 128, peaks), "b": ("y", 256, peaks)}, refs=refs)

    table = summarise(tmp_path, thresholds={"b": 0.8})["table"]["same_cell"]["all"]

    assert table["a"]["tp"] == 2
    assert (table["b"]["tp"], table["b"]["fn"]) == (1, 1)
    assert table["cfar"]["fn"] == 2


def test_summarise_refuses_a_folder_mixing_different_models(tmp_path):
    refs = [[56.50, 25.20, 1, True]]
    write_result(tmp_path, "s1", models={"a": ("x", 128, [])}, refs=refs)
    write_result(tmp_path, "s2", models={"a": ("other weights", 128, [])}, refs=refs)

    with pytest.raises(ValueError, match="other models"):
        summarise(tmp_path)


def test_summarise_refuses_passes_with_different_model_labels(tmp_path):
    refs = [[56.50, 25.20, 1, True]]
    write_result(tmp_path, "s1", models={"a": ("x", 128, [])}, refs=refs)
    write_result(tmp_path, "s2", models={"a": ("x", 128, []), "b": ("y", 128, [])}, refs=refs)

    with pytest.raises(ValueError, match="other models"):
        summarise(tmp_path)


def test_results_written_before_several_models_read_as_one_model_called_model(tmp_path):
    write_pass(tmp_path, "s1", refs=[[56.50, 25.20, 1, True]], model=[[56.50, 25.20, 0.9]])

    result = compare.load_result(tmp_path / "fujairah__s1.json")

    assert list(result["models"]) == ["model"]
    assert result["models"]["model"]["peaks"] == [[56.50, 25.20, 0.9]]
    assert result["models"]["model"]["sha256"] is None


def test_threshold_sweep_finds_the_threshold_with_the_best_f1(tmp_path):
    refs = [[56.50, 25.20, 1, True], [56.60, 25.30, 1, True]]
    peaks = [[56.50, 25.20, 0.9], [56.60, 25.30, 0.7], [56.70, 25.40, 0.3], [56.80, 25.50, 0.2]]
    write_result(tmp_path, "s1", models={"a": ("x", 128, peaks)}, refs=refs)

    sweep = summarise(tmp_path)["sweep"]["a"]["same_cell"]

    assert sweep["best_threshold"] == pytest.approx(0.35)
    assert sweep["best_f1"] == pytest.approx(1.0)
    assert sweep["by_threshold"]["0.25"]["f1"] == pytest.approx(0.8)


def test_bootstrap_of_identical_detectors_is_zero(tmp_path):
    for i in range(3):
        refs = [[56.50 + i, 25.20, 2, True], [56.60 + i, 25.30, 1, False]]
        write_result(
            tmp_path,
            f"s{i}",
            models={"a": ("x", 128, [[56.50 + i, 25.20, 0.9]])},
            refs=refs,
            cfar=[[56.50 + i, 25.20]],
        )

    boot = summarise(tmp_path, resamples=200)["bootstrap"]["a"]["neighbour_cells"]

    assert boot["f1"]["diff"] == pytest.approx(0.0)
    assert boot["f1"]["ci95"] == pytest.approx([0.0, 0.0])
    assert boot["recall_unmatched"]["diff"] == pytest.approx(0.0)


def test_bootstrap_difference_is_model_minus_cfar_with_an_interval_around_it(tmp_path):
    for i in range(8):
        refs = [[56.50 + i, 25.20, 1, True], [56.60 + i, 25.30, 1, False]]
        model = [[56.50 + i, 25.20, 0.9], [56.60 + i, 25.30, 0.9]]
        cfar = [[56.50 + i, 25.20]] if i % 2 else [[56.50 + i, 25.20], [56.60 + i, 25.30]]
        write_result(tmp_path, f"s{i}", models={"a": ("x", 128, model)}, refs=refs, cfar=cfar)

    boot = summarise(tmp_path, resamples=500)["bootstrap"]["a"]["same_cell"]

    # Model finds all 8 unmatched, CFAR 4 of 8: recall difference +0.5.
    assert boot["recall_unmatched"]["diff"] == pytest.approx(0.5)
    low, high = boot["recall_unmatched"]["ci95"]
    assert 0.0 < low <= 0.5 <= high <= 1.0
    assert boot["recall_ais_matched"]["diff"] == pytest.approx(0.0)


class FakeSession:
    """Stride-2 output with one bright cell per bright input pixel, as the trained model encodes."""

    def run(self, _names, feeds):
        image = feeds["image"]
        _, _, h, w = image.shape
        heat = np.full((1, 1, h // 2, w // 2), -10.0, dtype=np.float32)
        heat[0, 0] = np.where(image[0, 0, ::2, ::2] > 0.5, 10.0, -10.0)
        return heat, np.full((1, 2, h // 2, w // 2), -0.25, dtype=np.float32)


class OffsetScene:
    """lonlat = (col / 100, row / 100): scene pixel (row, col) -> lon col/100, lat row/100."""

    def lonlat(self, rows, cols):
        return np.asarray(cols) / 100.0, np.asarray(rows) / 100.0


def test_model_peaks_are_unmasked_inside_the_box_and_offset_by_the_window():
    image = np.zeros((2, 64, 64), dtype=np.float32)
    image[0, 10, 20] = image[0, 40, 40] = image[0, 50, 6] = 1.0
    masked = np.zeros((64, 64), dtype=bool)
    masked[40, 40] = True  # a masked target
    box = BBox(0.0, 0.0, 1.0, 0.45)  # the target at window row 50 lies at lat 0.5, outside

    peaks = compare.model_peaks(
        FakeSession(), image, masked, OffsetScene(), (0, 30), box, tile=64, overlap=0
    )

    # Cell (5, 10) with offset -0.25 decodes to (2 * 4.75 + 0.5, 2 * 9.75 + 0.5) = (10, 20);
    # scene column 20 + 30 -> lon 0.5, row 10 -> lat 0.1.
    assert [(round(lon, 4), round(lat, 4)) for lon, lat, _ in peaks] == [(0.5, 0.1)]
    assert peaks[0][2] > 0.99
