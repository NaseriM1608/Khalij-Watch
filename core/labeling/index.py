"""The combined dataset index: every chip of every area build, ready for training.

Area builds name chips per scene ("<stem>/d0000"), so a pass covering two
boxes yields the same ids in both; the index prefixes ids and shard paths with
the area. It also states what each chip may teach (``supervision``) and lists
the other known objects inside it, because a training loader that treats an
unlabelled detection near a chip's edge as sea would teach the model to miss
the very vessels that broadcast nothing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.labeling.chips import CHIP_SIZE

# What a chip teaches. POSITIVE: a vessel at the centre. IGNORE: something
# nobody identified is at the centre; teach nothing there. FULL_NEGATIVE: no
# target anywhere in the chip (sampled 1 km clear of every detection).
# CENTRE_NEGATIVE: the centre is not a vessel (a platform, a coastline, a fixed
# structure); the rest of the chip may hold vessels nobody detected, because
# CFAR masks the ground around platforms and along coasts.
POSITIVE = "positive"
IGNORE = "ignore"
FULL_NEGATIVE = "full_negative"
CENTRE_NEGATIVE = "centre_negative"

_FULL_NEGATIVE_SOURCES = ("sea", "clutter")

# Persistent structures: about 50 m bins, a spot being a bin and its neighbours.
PERSISTENCE_BIN_DEG = 0.0005
PERSISTENCE_MIN_SCENE_SHARE = 0.3
PERSISTENCE_MAX_LABELLED_SHARE = 0.05


def is_vessel(record: dict[str, Any]) -> bool:
    """AIS-matched by GFW, or seen by GFW's own detector without AIS.

    The second kind was "ignore" in kaggle-v1, and the model then missed
    vessels without AIS (recall 0.48 against CFAR's 0.77 on GFW detections
    without AIS).
    """
    if record["kind"] == "vessel":
        return True
    return record["kind"] == "unlabelled" and record["extra"].get("gfw_status") == "gfw_unmatched"


def supervision_for(record: dict[str, Any]) -> str:
    kind = record["kind"]
    if is_vessel(record):
        return POSITIVE
    if kind == "unlabelled":
        return IGNORE
    if record["source"] in _FULL_NEGATIVE_SOURCES:
        return FULL_NEGATIVE
    return CENTRE_NEGATIVE


def index_record(record: dict[str, Any], area: str, *, objects: list[list]) -> dict[str, Any]:
    """A new index entry for an area's chip record."""
    return {
        **record,
        "chip_id": f"{area}/{record['chip_id']}",
        "shard": f"{area}/{record['shard']}",
        "area": area,
        "supervision": supervision_for(record),
        "objects": objects,
    }


def as_persistent(record: dict[str, Any]) -> dict[str, Any]:
    """A copy of an unlabelled detection recast as a fixed structure: a centre negative."""
    return {
        **record,
        "kind": "negative",
        "source": "persistent",
        "extra": {**record["extra"], "relabelled_from": record["kind"]},
    }


def object_label(record: dict[str, Any]) -> str | None:
    """How a detection appears inside other chips, or None for chips of empty ground."""
    if is_vessel(record):
        return "vessel"
    if record["kind"] == "unlabelled":
        return "unlabelled"
    if record["kind"] == "negative" and record["source"] == "persistent":
        return "persistent"
    return None


def chip_objects(
    record: dict[str, Any], scene_detections: Iterable[dict[str, Any]], *, size: int = CHIP_SIZE
) -> list[list]:
    """[row offset, col offset, label] of every other known object inside the chip."""
    half = size // 2
    objects = []
    for other in scene_detections:
        if other is record or other["scene"] != record["scene"]:
            continue
        label = object_label(other)
        if label is None:
            continue
        d_row = other["row"] - record["row"]
        d_col = other["col"] - record["col"]
        if -half <= d_row < half and -half <= d_col < half:
            objects.append([d_row, d_col, label])
    return objects


@dataclass(frozen=True)
class PersistentSpots:
    """Bins covered by fixed structures that CFAR keeps detecting."""

    bins: frozenset[tuple[int, int]]
    count: int

    def __len__(self) -> int:
        return self.count

    def contains(self, record: dict[str, Any]) -> bool:
        return _bin(record) in self.bins


def persistent_spots(
    detections: Sequence[dict[str, Any]],
    *,
    scene_count: int,
    min_scene_share: float = PERSISTENCE_MIN_SCENE_SHARE,
    max_labelled_share: float = PERSISTENCE_MAX_LABELLED_SHARE,
) -> PersistentSpots:
    """Spots detected in at least ``min_scene_share`` of scenes and almost never labelled.

    A ship anchored for weeks is persistent too, which is why a spot where AIS
    identified a vessel more than ``max_labelled_share`` of the time is kept out.
    """
    by_bin: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in detections:
        if record["kind"] in ("vessel", "unlabelled"):
            by_bin[_bin(record)].append(record)

    claimed: set[tuple[int, int]] = set()
    spots = 0
    for key in sorted(by_bin, key=lambda k: -len(by_bin[k])):
        if key in claimed:
            continue
        around = [(key[0] + dx, key[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        members = [r for k in around for r in by_bin.get(k, [])]
        scenes = {r["scene"] for r in members}
        labelled = sum(r["kind"] == "vessel" for r in members)
        if len(scenes) >= min_scene_share * scene_count and labelled <= max_labelled_share * len(
            members
        ):
            claimed.update(around)
            spots += 1
    return PersistentSpots(frozenset(claimed), spots)


def _bin(record: dict[str, Any]) -> tuple[int, int]:
    return (
        round(record["lon"] / PERSISTENCE_BIN_DEG),
        round(record["lat"] / PERSISTENCE_BIN_DEG),
    )
