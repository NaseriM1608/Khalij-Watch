"""Tests for the chip dataset feeding the point detector (numpy only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.inference.contract import CHIP_SIZE, OUTPUT_SIZE
from core.inference.decode import decode_peaks
from training.data import ChipDataset, flip_sample, load_index, sampling_weight


def record(
    chip_id: str, index: int, *, supervision: str = "positive", split: str = "train", **extra
):
    base = {
        "chip_id": chip_id,
        "area": "a",
        "scene": "S.SAFE",
        "split": split,
        "shard": "a/shards/S.npz",
        "index": index,
        "row": 1000.0,
        "col": 2000.0,
        "kind": "vessel",
        "source": "gfw",
        "supervision": supervision,
        "objects": [],
    }
    return {**base, **extra}


@pytest.fixture
def dataset_root(tmp_path):
    chips = np.full((3, 2, CHIP_SIZE, CHIP_SIZE), 40, dtype=np.uint16)
    chips[0, :, 60:68, 62:66] = 1500  # a ship in chip 0
    (tmp_path / "a" / "shards").mkdir(parents=True)
    np.savez_compressed(
        tmp_path / "a" / "shards" / "S.npz",
        chips=chips,
        chip_ids=np.array(["S/d0", "S/n0", "S/d1"]),
    )
    records = [
        record("a/S/d0", 0),
        record("a/S/n0", 1, supervision="full_negative", kind="negative", source="sea"),
        record("a/S/d1", 2, split="val"),
    ]
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return tmp_path


def test_load_index_filters_by_split(dataset_root):
    assert [r["chip_id"] for r in load_index(dataset_root / "index.jsonl", "train")] == [
        "a/S/d0",
        "a/S/n0",
    ]
    assert len(load_index(dataset_root / "index.jsonl", "val")) == 1


def test_sample_holds_normalised_image_and_targets(dataset_root):
    records = load_index(dataset_root / "index.jsonl", "train")
    sample = ChipDataset(records, dataset_root, augment=False)[0]

    assert sample["image"].shape == (2, CHIP_SIZE, CHIP_SIZE)
    assert sample["image"].dtype == np.float32
    assert sample["image"].min() > 0.0
    assert sample["image"].max() < 1.0
    assert sample["heatmap"].shape == (1, OUTPUT_SIZE, OUTPUT_SIZE)
    assert sample["weight"].shape == (1, OUTPUT_SIZE, OUTPUT_SIZE)
    assert sample["offset"].shape == (2, OUTPUT_SIZE, OUTPUT_SIZE)
    assert sample["heatmap"].max() == pytest.approx(1.0)


def test_negative_sample_has_empty_heatmap(dataset_root):
    records = load_index(dataset_root / "index.jsonl", "train")

    assert ChipDataset(records, dataset_root, augment=False)[1]["heatmap"].max() == 0


def test_flipping_moves_image_and_targets_together():
    """After any flip, the decoded peak must sit on the flipped ship."""
    rng = np.random.default_rng(0)
    image = rng.random((2, CHIP_SIZE, CHIP_SIZE)).astype(np.float32)
    from core.labeling.targets import encode_targets

    targets = encode_targets(
        {
            "supervision": "positive",
            "row": 1000.3,
            "col": 2000.2,
            "objects": [[20.0, -35.0, "vessel"]],
        }
    )
    sample = {
        "image": image,
        "heatmap": targets["heatmap"][None],
        "weight": targets["weight"][None],
        "offset": targets["offset"],
        "offset_mask": targets["offset_mask"][None],
    }
    before = sorted(decode_peaks(sample["heatmap"][0], sample["offset"], threshold=0.5))

    flipped = flip_sample(sample, vertical=True, horizontal=True)
    after = sorted(decode_peaks(flipped["heatmap"][0], flipped["offset"], threshold=0.5))

    expected = sorted((CHIP_SIZE - 1 - r, CHIP_SIZE - 1 - c, s) for r, c, s in before)
    np.testing.assert_allclose(np.array(after), np.array(expected), atol=1e-4)
    assert np.array_equal(flipped["image"], image[:, ::-1, ::-1])


def test_random_crop_moves_the_ship_off_centre_and_keeps_targets_aligned():
    from core.labeling.targets import encode_targets
    from training.data import random_crop

    image = np.zeros((2, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    targets = encode_targets(
        {"supervision": "positive", "row": 1000.0, "col": 2000.0, "objects": []}
    )
    sample = {
        "image": image,
        "heatmap": targets["heatmap"][None],
        "weight": targets["weight"][None],
        "offset": targets["offset"],
        "offset_mask": targets["offset_mask"][None],
    }

    cropped = random_crop(sample, 96, top=10, left=30)

    assert cropped["image"].shape == (2, 96, 96)
    assert cropped["heatmap"].shape == (1, 48, 48)
    ((row, col, _),) = decode_peaks(cropped["heatmap"][0], cropped["offset"], threshold=0.5)
    assert (row, col) == pytest.approx((64.0 - 10, 64.0 - 30))


def test_augmented_samples_are_cropped_for_jitter(dataset_root):
    records = load_index(dataset_root / "index.jsonl", "train")
    sample = ChipDataset(records, dataset_root, augment=True, seed=1, crop=96)[0]

    assert sample["image"].shape == (2, 96, 96)
    assert sample["heatmap"].shape == (1, 48, 48)
    assert sample["heatmap"].max() == pytest.approx(1.0)


def test_augmented_samples_keep_shapes_and_are_unclipped(dataset_root):
    records = load_index(dataset_root / "index.jsonl", "train")
    dataset = ChipDataset(records, dataset_root, augment=True, seed=1)

    sample = dataset[0]

    assert sample["image"].shape == (2, CHIP_SIZE, CHIP_SIZE)
    assert sample["heatmap"].max() == pytest.approx(1.0)


def test_sampling_weights_favour_positives_and_hard_negatives_over_easy_sea():
    positive = record("p", 0)
    sea = record("s", 0, supervision="full_negative", kind="negative", source="sea")
    coast = record("c", 0, supervision="centre_negative", kind="negative", source="coast")
    unknown = record("u", 0, supervision="ignore", kind="unlabelled")

    assert sampling_weight(positive) == sampling_weight(coast) > sampling_weight(sea)
    assert sampling_weight(sea) > sampling_weight(unknown) > 0
