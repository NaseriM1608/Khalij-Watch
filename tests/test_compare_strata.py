"""Tests for the sensitivity strata of a detector comparison against GFW.

Fixed before the Iranian-coast results were read: (1) only cells at least
1 km from land, GFW's convention (the comparison itself masks 150 m), applied
to detections and GFW references alike; (2) only passes whose datatake has no
frame the model trained or was chosen on.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from core.sar.coastline import LandPolygon
from scripts import compare_cfar_model as compare
from scripts import compare_strata as strata

SCENE = "S1A_IW_GRDH_1SDV_20240114T141727_20240114T141752_052096_064C2B_1A2B_COG.SAFE"
SIBLING = "S1A_IW_GRDH_1SDV_20240114T141702_20240114T141727_052096_064C2B_9F9F_COG.SAFE"
OTHER = "S1A_IW_GRDH_1SDV_20240203T141726_20240203T141751_052387_065661_77AA_COG.SAFE"
# Land west of 56.40E (a straight north-south coast).
LAND = [LandPolygon(np.array([[55.0, 24.0], [56.40, 24.0], [56.40, 27.0], [55.0, 27.0]]))]


def test_a_datatake_is_the_satellite_and_datatake_id_of_a_frame():
    assert strata.datatake(SCENE) == ("S1A", "064C2B")
    assert strata.datatake(SCENE) == strata.datatake(SIBLING)
    assert strata.datatake(OTHER) != strata.datatake(SCENE)


def test_seen_datatakes_are_those_of_the_named_splits(tmp_path):
    index = tmp_path / "index.jsonl"
    rows = [("train", SIBLING), ("test", OTHER)]
    index.write_text(
        "".join(json.dumps({"split": s, "area": "x", "scene": n}) + "\n" for s, n in rows),
        encoding="utf-8",
    )
    assert strata.seen_datatakes(index, {"train", "val"}) == {("S1A", "064C2B")}


def result(cfar, peaks, refs, scene=SCENE, area="fujairah"):
    return {
        "area": area,
        "scene": scene,
        "split": "x",
        "cfar": cfar,
        "models": {"run2": {"sha256": "s", "tile": 256, "overlap": 64, "peaks": peaks}},
        "refs": refs,
        "gfw_in_box": sum(r[2] for r in refs),
    }


def test_cells_near_land_are_left_out_for_detections_and_references_alike():
    # 56.41E is about 1 km from the coast at 56.40E; 56.45E about 5 km.
    near, far = (56.41, 25.20), (56.45, 25.20)
    r = result(
        cfar=[list(near), list(far)],
        peaks=[[*near, 0.9], [*far, 0.8]],
        refs=[[*near, 2, True], [*far, 1, False]],
    )

    kept = strata.away_from_land(r, LAND, 1500.0)

    assert kept["cfar"] == [list(far)]
    assert kept["models"]["run2"]["peaks"] == [[*far, 0.8]]
    assert kept["refs"] == [[*far, 1, False]]
    assert kept["gfw_in_box"] == r["gfw_in_box"]  # whether GFW covered the pass is unchanged
    assert r["cfar"] == [list(near), list(far)]  # the input is not modified


def test_a_detection_is_judged_by_its_cell_centre_like_the_reference_it_may_match():
    # The detection (1.52 km from land) lies in the cell centred on 56.42E (2.01 km).
    cell_centre_far_enough = strata.away_from_land(
        result(cfar=[[56.4151, 25.20]], peaks=[], refs=[[56.42, 25.20, 1, True]]), LAND, 1600.0
    )
    assert cell_centre_far_enough["cfar"] == [[56.4151, 25.20]]
    assert cell_centre_far_enough["refs"] == [[56.42, 25.20, 1, True]]


def test_main_summarises_each_stratum_with_the_comparisons_own_scoring(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "compare_x"
    out.mkdir()
    far = [56.45, 25.20]
    for scene in (SCENE, OTHER):
        path = compare.result_path(out, "fujairah", scene)
        path.write_text(json.dumps(result([far], [[*far, 0.9]], [[*far, 1, True]], scene)))
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"split": "train", "area": "x", "scene": SIBLING}) + "\n")
    monkeypatch.setattr(strata, "land_near", lambda _box: LAND)
    work = tmp_path / "work"
    argv = ["strata", "--out", str(out), "--work", str(work), "--shore-m", "1000"]
    monkeypatch.setattr(sys, "argv", [*argv, "--index", str(index), "--threshold", "run2=0.6"])

    assert strata.main() == 0

    report = json.loads((work / "strata.json").read_text(encoding="utf-8"))
    assert not (out / "strata.json").exists()  # compare_cfar_model reads every *.json in --out
    assert report["all"]["scenes"] == 2
    assert report["shore_1000m"]["scenes"] == 2
    assert report["unseen_datatakes"]["scenes"] == 1
    assert report["unseen_datatakes_shore_1000m"]["scenes"] == 1
    row = report["shore_1000m"]["table"]["same_cell"]["all"]["run2"]
    assert row["tp"] == 2 and row["fp"] == 0
    assert "shore_1000m" in capsys.readouterr().out


def test_main_refuses_a_folder_without_results(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["strata", "--out", str(tmp_path), "--shore-m", "1000"])
    with pytest.raises(SystemExit):
        strata.main()


def test_detections_at_persistent_spots_are_left_out_but_gfw_references_are_kept():
    """Hiding a long-moored ship GFW reports must cost recall, so references stay."""
    from core.persistence import Spot

    spot = Spot(56.45, 25.20, "fujairah", 5, 10, "2024-01-01", "2025-11-03")
    r = result(
        cfar=[[56.4505, 25.2005], [56.50, 25.25]],
        peaks=[[56.4505, 25.2005, 0.9], [56.50, 25.25, 0.8]],
        refs=[[56.45, 25.20, 1, False]],
    )

    kept = strata.without_spots(r, [spot])

    assert kept["cfar"] == [[56.50, 25.25]]
    assert kept["models"]["run2"]["peaks"] == [[56.50, 25.25, 0.8]]
    assert kept["refs"] == r["refs"]


def test_main_adds_the_persistent_spot_strata_when_given_spots(tmp_path, monkeypatch):
    from core.persistence import Spot, save_spots

    out = tmp_path / "compare_x"
    out.mkdir()
    at_spot, elsewhere = [56.45, 25.20], [56.47, 25.22]
    path = compare.result_path(out, "fujairah", OTHER)
    path.write_text(
        json.dumps(
            result(
                [at_spot, elsewhere],
                [[*at_spot, 0.9], [*elsewhere, 0.9]],
                [[*elsewhere, 1, True]],
                OTHER,
            )
        )
    )
    spots = tmp_path / "spots.json"
    save_spots(spots, [Spot(*at_spot, "fujairah", 5, 10, "2024-01-01", "2025-11-03")], {})
    monkeypatch.setattr(strata, "land_near", lambda _box: LAND)
    work = tmp_path / "work"
    argv = ["strata", "--out", str(out), "--work", str(work), "--shore-m", "1000"]
    monkeypatch.setattr(sys, "argv", [*argv, "--spots", str(spots), "--threshold", "run2=0.6"])

    assert strata.main() == 0

    report = json.loads((work / "strata.json").read_text(encoding="utf-8"))
    assert report["all"]["table"]["same_cell"]["all"]["run2"]["fp"] == 1
    assert report["persistent_masked"]["table"]["same_cell"]["all"]["run2"]["fp"] == 0
    assert "persistent_masked_shore_1000m" in report
