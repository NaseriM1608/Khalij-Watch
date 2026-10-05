"""Tests for choosing the passes a comparison runs on, by a rule fixed before any result.

A held-out check (the product model on the Iranian coast) must not compare
scenes the model saw in training or was chosen on, and the passes must be
picked by a stated rule, not by looking at results.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from types import SimpleNamespace

import pytest

from core.areas import IRAN_AREAS, box_around
from scripts import compare_cfar_model as compare
from scripts import list_pairs

A1, A2, A3, A4 = (f"S1A_IW_GRDH_1SDV_2024{m:02d}01T021200_X{m}_COG.SAFE" for m in (1, 2, 3, 4))
C1 = "S1C_IW_GRDH_1SDV_20260910T021440_Y_COG.SAFE"
D1 = "S1D_IW_GRDH_1SDV_20260912T020650_Z_COG.SAFE"


def test_scenes_the_model_trained_or_was_chosen_on_are_left_out():
    pairs, _ = list_pairs.choose(
        {"kish": [A1, A2, A3]}, satellites={"S1A"}, exclude={A2}, per_area=0
    )
    assert pairs == [("kish", A1), ("kish", A3)]


def test_only_the_chosen_satellites_are_kept():
    found = {"jask": [A1, C1, D1]}
    pairs, _ = list_pairs.choose(found, satellites={"S1C", "S1D"}, exclude=set(), per_area=0)
    assert pairs == [("jask", C1), ("jask", D1)]


def test_up_to_per_area_passes_are_spread_through_each_box_like_the_comparisons():
    found = {"kish": [A4, A1, A3, A2], "jask": [A1]}
    pairs, _ = list_pairs.choose(found, satellites={"S1A"}, exclude=set(), per_area=2)
    assert pairs == compare.spread([("kish", s) for s in found["kish"]] + [("jask", A1)], 2)
    assert pairs == [("jask", A1), ("kish", A1), ("kish", A3)]


def test_each_box_reports_why_passes_were_left_out():
    found = {"kish": [A1, A2, A3, C1], "jask": []}
    _, counts = list_pairs.choose(found, satellites={"S1A"}, exclude={A3, C1}, per_area=1)
    assert counts == {
        "jask": {"found": 0, "other_satellites": 0, "seen": 0, "kept": 0, "chosen": 0},
        "kish": {"found": 4, "other_satellites": 1, "seen": 1, "kept": 2, "chosen": 1},
    }


def test_seen_scenes_are_the_scenes_of_the_named_index_splits(tmp_path):
    index = tmp_path / "index.jsonl"
    rows = [("train", A1), ("train", A1), ("val", A2), ("test", A3)]
    index.write_text(
        "".join(json.dumps({"split": s, "area": "fujairah", "scene": n}) + "\n" for s, n in rows),
        encoding="utf-8",
    )
    assert list_pairs.seen_scenes(index, {"train", "val"}) == {A1, A2}
    assert list_pairs.seen_scenes(index, set()) == set()


def test_the_written_file_states_its_rule_and_reads_back_as_the_pairs(tmp_path):
    out = tmp_path / "pairs_iran.txt"
    pairs = [("bandar_abbas", A1), ("jask", A2)]

    list_pairs.write_pairs(out, pairs, ["rule: every pass", "bandar_abbas found 1"])

    text = out.read_text(encoding="utf-8")
    assert text.startswith("# rule: every pass\n# bandar_abbas found 1\n")
    assert compare.read_pairs(out) == pairs


def test_main_searches_each_iranian_box_once_and_writes_the_chosen_pairs(
    tmp_path, monkeypatch, capsys
):
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"split": "train", "area": "x", "scene": A2}) + "\n")
    searched = []

    def passes_over(box, since, until):
        searched.append((box, since, until))
        return [SimpleNamespace(name=n) for n in (A1, A2, C1)]

    monkeypatch.setitem(
        sys.modules, "scripts.build_dataset", SimpleNamespace(passes_over=passes_over)
    )
    out = tmp_path / "pairs.txt"
    argv = ["list_pairs", "--since", "2024-01-01", "--until", "2026-01-01", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--index", str(index)])

    assert list_pairs.main() == 0

    assert len(searched) == 7
    assert {box for box, _, _ in searched} == {box_around(*spec) for spec in IRAN_AREAS.values()}
    assert {(s, u) for _, s, u in searched} == {
        (dt.datetime(2024, 1, 1, tzinfo=dt.UTC), dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    }
    assert {a for a, _ in compare.read_pairs(out)} == {
        "bandar_abbas",
        "larak_hormuz",
        "kish",
        "assaluyeh",
        "bushehr",
        "kharg",
        "jask",
    }
    assert {s for _, s in compare.read_pairs(out)} == {A1}
    assert "kish" in capsys.readouterr().out


def test_main_refuses_an_empty_date_range(tmp_path, monkeypatch):
    out = tmp_path / "pairs.txt"
    argv = ["list_pairs", "--since", "2026-01-01", "--until", "2026-01-01", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        list_pairs.main()
    assert not out.exists()
