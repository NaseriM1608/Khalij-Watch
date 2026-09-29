"""Build the combined training index over every area under data/datasets.

Reads each area's ``manifest.jsonl`` and its mined hard negatives
(``hardneg/records/*.jsonl``), recasts unlabelled detections at persistent
fixed structures as negatives, prefixes ids and shard paths with the area,
lists the known objects inside every chip, and writes ``index.jsonl`` plus
``index_meta.json`` (counts and the rules used) to the root. With ``--verify``
it also opens every shard and checks that each record's index holds its chip.

Run:
    uv run python scripts/build_index.py [--root data/datasets] [--verify]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.labeling.index import (  # noqa: E402
    PERSISTENCE_MAX_LABELLED_SHARE,
    PERSISTENCE_MIN_SCENE_SHARE,
    as_persistent,
    chip_objects,
    index_record,
    object_label,
    persistent_spots,
)

INDEX_VERSION = "khalij-watch-index-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_area(area_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """An area's records, persistent structures recast; returns them and the spot count."""
    built = read_jsonl(area_dir / "manifest.jsonl")
    mined = [
        r for p in sorted((area_dir / "hardneg" / "records").glob("*.jsonl")) for r in read_jsonl(p)
    ]
    spots = persistent_spots(built, scene_count=len({r["scene"] for r in built}))
    recast = [
        as_persistent(r) if r["kind"] == "unlabelled" and spots.contains(r) else r for r in built
    ]
    return recast + mined, len(spots)


def build(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    areas = sorted(p.parent.name for p in root.glob("*/manifest.jsonl"))
    by_area: dict[str, list[dict[str, Any]]] = {}
    spot_counts: dict[str, int] = {}
    for area in areas:
        by_area[area], spot_counts[area] = load_area(root / area)

    # Neighbours come from every area: boxes do not overlap, but chips reach past them.
    objects_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for records in by_area.values():
        for r in records:
            if object_label(r) is not None:
                objects_by_scene[r["scene"]].append(r)

    index = [
        index_record(r, area, objects=chip_objects(r, objects_by_scene[r["scene"]]))
        for area in areas
        for r in by_area[area]
    ]
    ids = Counter(r["chip_id"] for r in index)
    duplicates = [i for i, n in ids.items() if n > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate chip ids, e.g. {duplicates[:3]}")
    return index, meta_for(index, areas, spot_counts)


def meta_for(
    index: list[dict[str, Any]], areas: list[str], spots: dict[str, int]
) -> dict[str, Any]:
    def table(key) -> dict[str, int]:
        return dict(sorted(Counter(key(r) for r in index).items()))

    return {
        "version": INDEX_VERSION,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "areas": areas,
        "chips": len(index),
        "scenes": len({r["scene"] for r in index}),
        "by_split_supervision": table(lambda r: f"{r['split']}/{r['supervision']}"),
        "by_kind_source": table(lambda r: f"{r['kind']}/{r['source']}"),
        "by_area_supervision": table(lambda r: f"{r['area']}/{r['supervision']}"),
        "persistent_spots": spots,
        "rules": {
            "persistent": (
                f"unlabelled detections at spots detected in >= {PERSISTENCE_MIN_SCENE_SHARE:.0%} "
                f"of an area's scenes, labelled <= {PERSISTENCE_MAX_LABELLED_SHARE:.0%} of the time"
            ),
            "supervision": {
                "positive": "vessel at the centre (GFW AIS match)",
                "ignore": "unidentified detection at the centre; may be a dark vessel",
                "full_negative": "no target anywhere in the chip (sea, clutter)",
                "centre_negative": "centre is not a vessel (platform, coast, persistent)",
            },
            "objects": "[row offset, col offset, label] of other known objects in the chip",
        },
        "caveats": [
            "vessel_type is unreliable: GFW folded tankers into OTHER between 2026-09-25 and "
            "2026-09-27, so most labels fetched later carry no coarse class",
            "detections outside an area's box but inside its read window are not listed in objects",
            "chip pixel (64, 64) is round(row, col); objects are float offsets from row, col",
        ],
    }


def verify(root: Path, index: list[dict[str, Any]]) -> int:
    """Open every shard and check each record's index holds its chip; returns problems."""
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in index:
        by_shard[r["shard"]].append(r)
    problems = 0
    for shard, records in by_shard.items():
        path = root / shard
        if not path.is_file():
            problems += len(records)
            print(f"missing shard {shard} ({len(records)} records)")
            continue
        with np.load(path) as data:
            ids = data["chip_ids"]
        for r in records:
            area_prefix = f"{r['area']}/"
            if r["index"] >= len(ids) or ids[r["index"]] != r["chip_id"].removeprefix(area_prefix):
                problems += 1
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data") / "datasets")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    index, meta = build(args.root)
    if args.verify:
        meta["verify_problems"] = verify(args.root, index)
        print(f"verify: {meta['verify_problems']} problems")
    index.sort(key=lambda r: (r["area"], r["scene"], r["chip_id"]))
    (args.root / "index.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in index), encoding="utf-8"
    )
    (args.root / "index_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {k: meta[k] for k in ("chips", "scenes", "by_kind_source", "persistent_spots")},
            indent=1,
        )
    )
    return 1 if meta.get("verify_problems") else 0


if __name__ == "__main__":
    raise SystemExit(main())
