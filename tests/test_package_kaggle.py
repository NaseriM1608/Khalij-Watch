"""Tests for staging the chip dataset and training code for a private Kaggle dataset."""

from __future__ import annotations

import json

import numpy as np

from scripts.package_kaggle import dataset_metadata, stage


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
