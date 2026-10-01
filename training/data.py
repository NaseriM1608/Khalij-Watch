"""Chip dataset: index records to normalised chips and dense targets (numpy only).

PyTorch's DataLoader accepts any indexable object, so this stays free of
torch and is tested like the rest of the library. Augmentation follows the
SAR-safe list in PLAN.md section 7.1b and never clips: brightness gain per
band on raw amplitude, Gaussian noise after normalisation, and flips.
Rotation, scale and elastic warps are left for later ablations.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from core.inference.contract import normalise_chip
from core.labeling.targets import encode_targets

# Relative sampling weight by what a chip teaches. Easy open sea is plentiful
# (about three quarters of all negatives) and is sampled down; unidentified
# detections teach only their surroundings.
_WEIGHTS = {
    "positive": 1.0,
    "centre_negative": 1.0,
    "full_negative": 1.0,
    "ignore": 0.1,
}
EASY_SEA_WEIGHT = 0.15

GAIN_RANGE = (0.8, 1.25)
NOISE_SIGMA = 0.01
SHARD_CACHE_SIZE = 32


def load_index(path: Path, split: str) -> list[dict[str, Any]]:
    """Index records of one split."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["split"] == split:
                    records.append(record)
    return records


def sampling_weight(record: dict[str, Any]) -> float:
    if record["supervision"] == "full_negative" and record["source"] == "sea":
        return EASY_SEA_WEIGHT
    return _WEIGHTS[record["supervision"]]


def flip_sample(
    sample: dict[str, np.ndarray], *, vertical: bool, horizontal: bool
) -> dict[str, np.ndarray]:
    """A new sample flipped along rows and/or columns; offsets change sign with it."""
    out = dict(sample)
    for flip, axis in ((vertical, -2), (horizontal, -1)):
        if not flip:
            continue
        for key in ("image", "heatmap", "weight", "offset", "offset_mask"):
            out[key] = np.flip(out[key], axis=axis)
        component = 0 if axis == -2 else 1
        offset = out["offset"].copy()
        offset[component] = -offset[component]
        out["offset"] = offset
    return {key: np.ascontiguousarray(value) for key, value in out.items()}


def random_crop(
    sample: dict[str, np.ndarray], size: int, *, top: int, left: int
) -> dict[str, np.ndarray]:
    """A size x size window of a sample at even (top, left), targets cut to match."""
    if top % 2 or left % 2 or size % 2:
        raise ValueError("crops must start and end on even pixels to keep the stride-2 grid")
    out_top, out_left, out_size = top // 2, left // 2, size // 2
    out = {"image": sample["image"][:, top : top + size, left : left + size]}
    for key in ("heatmap", "weight", "offset", "offset_mask"):
        out[key] = sample[key][:, out_top : out_top + out_size, out_left : out_left + out_size]
    return {key: np.ascontiguousarray(value) for key, value in out.items()}


class ChipDataset:
    """Map-style dataset over index records; shards load lazily per worker."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        root: Path,
        *,
        augment: bool,
        seed: int = 0,
        crop: int = 0,
    ) -> None:
        self.records = records
        self.root = Path(root)
        self.augment = augment
        self.seed = seed
        # Random crops while augmenting, so a vessel is not always at the centre.
        self.crop = crop
        self._shards: OrderedDict[str, np.ndarray] = OrderedDict()
        self._draws = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        record = self.records[i]
        raw = self._chip(record["shard"], record["index"]).astype(np.float32)
        rng = np.random.default_rng((self.seed, i, self._next_draw())) if self.augment else None
        if rng is not None:
            raw = raw * rng.uniform(*GAIN_RANGE, size=(raw.shape[0], 1, 1)).astype(np.float32)
        image = normalise_chip(raw)
        if rng is not None:
            image = image + rng.normal(0.0, NOISE_SIGMA, size=image.shape).astype(np.float32)
        targets = encode_targets(record)
        sample = {
            "image": image.astype(np.float32),
            "heatmap": targets["heatmap"][None],
            "weight": targets["weight"][None],
            "offset": targets["offset"],
            "offset_mask": targets["offset_mask"][None],
        }
        if rng is not None:
            sample = flip_sample(
                sample, vertical=bool(rng.integers(2)), horizontal=bool(rng.integers(2))
            )
            if self.crop:
                room = (image.shape[-1] - self.crop) // 2
                top, left = 2 * int(rng.integers(room + 1)), 2 * int(rng.integers(room + 1))
                sample = random_crop(sample, self.crop, top=top, left=left)
        return sample

    def _next_draw(self) -> int:
        self._draws += 1
        return self._draws

    def _chip(self, shard: str, index: int) -> np.ndarray:
        if shard not in self._shards:
            with np.load(self.root / shard) as data:
                self._shards[shard] = data["chips"]
            if len(self._shards) > SHARD_CACHE_SIZE:
                self._shards.popitem(last=False)
        self._shards.move_to_end(shard)
        return self._shards[shard][index]
