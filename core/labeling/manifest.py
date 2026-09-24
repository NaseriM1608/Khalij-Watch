"""Dataset manifest: one record per chip, and the scene-level split.

Split by scene, never by chip. Chips cut from one scene share its sea state,
calibration and neighbouring vessels, so a random split would leak them across
train and test and inflate every metric. Splitting by scene does not remove
all leakage: ships at anchor appear in consecutive passes, which is why the
evaluation protocol also reports results by location (PLAN.md section 12).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
# Changing the salt reshuffles every scene; bump it only for a new dataset version.
SPLIT_SALT = "khalij-watch-v1"

# vessel: a detection with an AIS-derived label. negative: known not a vessel
# (open sea, a platform, land). unlabelled: a detection nobody has identified,
# which may well be a vessel broadcasting nothing, so it is never a negative.
KINDS = ("vessel", "negative", "unlabelled")


def split_for(scene: str, *, salt: str = SPLIT_SALT) -> str:
    """The split a scene belongs to: stable across runs and machines."""
    digest = hashlib.sha256(f"{salt}:{scene}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name, share in zip(SPLITS, SPLIT_FRACTIONS, strict=True):
        cumulative += share
        if fraction < cumulative:
            return name
    return SPLITS[-1]


@dataclass(frozen=True)
class ChipRecord:
    """One chip: where it came from, where it is stored, and what it shows."""

    chip_id: str
    scene: str
    split: str
    shard: str
    index: int
    lon: float
    lat: float
    row: float
    col: float
    kind: str
    source: str
    mmsi: int | None = None
    length_m: float | None = None
    vessel_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {KINDS}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; expected one of {SPLITS}")


def records_to_jsonl(records: Sequence[ChipRecord]) -> str:
    return "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)


def records_from_jsonl(text: str) -> list[ChipRecord]:
    return [ChipRecord(**json.loads(line)) for line in text.splitlines() if line.strip()]
