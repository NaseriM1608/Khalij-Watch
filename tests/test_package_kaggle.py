"""Tests for staging the chip dataset and training code for a private Kaggle dataset."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts.package_kaggle import dataset_metadata, main, stage


def write_index(root, shards):
    root.mkdir(parents=True, exist_ok=True)
    records = [{"chip_id": f"c{i}", "shard": s, "index": 0} for i, s in enumerate(shards)]
    (root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "index_meta.json").write_text("{}")
    for shard in set(shards):
        path = root / shard
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chips=np.zeros((1, 2, 4, 4), dtype=np.uint16))


def test_stage_links_every_referenced_shard_once_and_nothing_else(tmp_path):
    root, out = tmp_path / "data", tmp_path / "stage"
    write_index(root, ["a/shards/S.npz", "a/shards/S.npz", "b/hardneg/shards/T.npz"])
    (root / "a" / "shards" / "unused.npz").write_bytes(b"x")

    report = stage(root, out, code_dirs=[])

    assert (out / "a" / "shards" / "S.npz").is_file()
    assert (out / "b" / "hardneg" / "shards" / "T.npz").is_file()
    assert not (out / "a" / "shards" / "unused.npz").exists()
    assert (out / "index.jsonl").is_file()
    assert report["shards"] == 2


def test_stage_copies_code_without_caches(tmp_path):
    root, out, code = tmp_path / "data", tmp_path / "stage", tmp_path / "training"
    write_index(root, [])
    (code / "__pycache__").mkdir(parents=True)
    (code / "train.py").write_text("print('hi')")
    (code / "__pycache__" / "train.cpython-312.pyc").write_bytes(b"x")

    stage(root, out, code_dirs=[code])

    assert (out / "code" / "training" / "train.py").is_file()
    assert not (out / "code" / "training" / "__pycache__").exists()


def test_index_only_staging_leaves_the_shards_out(tmp_path):
    root, out = tmp_path / "data", tmp_path / "stage"
    write_index(root, ["a/shards/S.npz"])

    report = stage(root, out, code_dirs=[], shards=False)

    assert (out / "index.jsonl").is_file()
    assert not (out / "a").exists()
    assert report["shards"] == 0


def test_metadata_is_private_licence_other_and_named():
    meta = dataset_metadata("someone/khalij-watch-chips")

    assert meta["id"] == "someone/khalij-watch-chips"
    assert meta["licenses"] == [{"name": "other"}]
    assert "Global Fishing Watch" in meta["description"]
    assert "Copernicus" in meta["description"]


def test_chips_description_counts_gfw_detections_without_ais_as_vessels():
    meta = dataset_metadata("someone/khalij-watch-chips")

    assert "without AIS" in meta["description"]


def test_index_only_metadata_does_not_claim_chips_and_fits_kaggle_limits():
    chips = dataset_metadata("someone/khalij-watch-chips")
    meta = dataset_metadata("someone/khalij-watch-index", index_only=True)

    assert meta["id"] == "someone/khalij-watch-index"
    assert meta["licenses"] == [{"name": "other"}]
    assert "chips" not in meta["title"]
    assert meta["title"] != chips["title"]
    assert meta["subtitle"] != chips["subtitle"]
    assert meta["description"] != chips["description"]
    assert "Global Fishing Watch" in meta["description"]
    assert "without AIS" in meta["description"]
    for m in (chips, meta):
        assert 20 <= len(m["subtitle"]) <= 80


def test_index_only_refuses_a_folder_that_holds_shards(tmp_path):
    root, out = tmp_path / "data", tmp_path / "stage"
    write_index(root, ["a/shards/S.npz"])
    stage(root, out, code_dirs=[])
    (out / "index.jsonl").write_text("old")

    with pytest.raises(ValueError, match="shards"):
        stage(root, out, code_dirs=[], shards=False)

    assert (out / "index.jsonl").read_text() == "old"


def test_index_only_main_without_out_stages_into_its_own_folder(tmp_path, monkeypatch):
    root = tmp_path / "data" / "datasets"
    write_index(root, ["a/shards/S.npz"])
    chips = tmp_path / "data" / "kaggle" / "khalij-watch-chips"
    chips.mkdir(parents=True)
    (chips / "dataset-metadata.json").write_text("chips")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["package_kaggle.py", "--index-only", "--dataset-id", "someone/khalij-watch-index"],
    )

    assert main() == 0

    index = tmp_path / "data" / "kaggle" / "khalij-watch-index"
    assert (index / "index.jsonl").is_file()
    meta = json.loads((index / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/khalij-watch-index"
    assert (chips / "dataset-metadata.json").read_text() == "chips"
    assert not (chips / "index.jsonl").exists()
