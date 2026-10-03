"""Stage the chip dataset and training code for upload as a PRIVATE Kaggle dataset.

Builds a folder holding index.jsonl, index_meta.json, every shard the index
references (hard-linked, so nothing is copied on the same drive), the code
that training imports (core/ and training/, without caches), and Kaggle's
dataset-metadata.json. Uploading is a separate, explicit step:

    kaggle datasets create -p data/kaggle/khalij-watch-chips --dir-mode tar

(``create`` makes a private dataset unless ``--public`` is given.) Labels come
from Global Fishing Watch for non-commercial use, so the dataset must stay
private.

Run:
    uv run python scripts/package_kaggle.py --dataset-id <kaggle-user>/khalij-watch-chips
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    "Sentinel-1 VV+VH chips (128x128, uint16 amplitude) of vessels and negatives in the "
    "Persian Gulf and Gulf of Oman, 2024-2025, with a training index. Vessel labels: Global "
    "Fishing Watch SAR vessel detections, with or without AIS (non-commercial use only). "
    "Contains modified Copernicus Sentinel data. Platform positions: Offshore Platform Dataset "
    "(CC BY 4.0). Coastline: OpenStreetMap contributors (ODbL). Private research dataset."
)

INDEX_DESCRIPTION = (
    "Training index (index.jsonl, index_meta.json) and training code for the khalij-watch "
    "Sentinel-1 vessel chips of the Persian Gulf and Gulf of Oman; the chips themselves are "
    "in the separate chip dataset. Vessel labels: Global Fishing Watch SAR vessel detections, "
    "with or without AIS (non-commercial use only). Contains modified Copernicus Sentinel "
    "data. Platform positions: Offshore Platform Dataset (CC BY 4.0). Coastline: "
    "OpenStreetMap contributors (ODbL). Private research dataset."
)


def dataset_metadata(dataset_id: str, *, index_only: bool = False) -> dict[str, Any]:
    if index_only:
        return {
            "id": dataset_id,
            "title": "khalij-watch Gulf SAR training index",
            "subtitle": "Training index and code for the khalij-watch chip dataset",
            "description": INDEX_DESCRIPTION,
            "licenses": [{"name": "other"}],
        }
    return {
        "id": dataset_id,
        "title": "khalij-watch Gulf SAR vessel chips",
        "subtitle": "Sentinel-1 chips with AIS-derived labels and hard negatives",
        "description": DESCRIPTION,
        "licenses": [{"name": "other"}],
    }


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def stage(root: Path, out: Path, *, code_dirs: list[Path], shards: bool = True) -> dict[str, Any]:
    """Assemble the upload folder; returns counts and total bytes.

    With ``shards=False`` only the index and code are staged: a new labelling
    of the same chips, uploaded beside the existing chip dataset. It refuses a
    folder that already holds shards, so the chip dataset's index is never replaced.
    """
    if not shards and out.exists() and any(p.is_dir() for p in out.rglob("shards")):
        raise ValueError(f"{out} holds shards; stage the index into its own folder")
    out.mkdir(parents=True, exist_ok=True)
    referenced: set[str] = set()
    with (root / "index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and shards:
                referenced.add(json.loads(line)["shard"])
    for name in ("index.jsonl", "index_meta.json"):
        shutil.copy2(root / name, out / name)
    for shard in sorted(referenced):
        _link_or_copy(root / shard, out / shard)
    for code in code_dirs:
        target = out / "code" / code.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(code, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {"shards": len(referenced), "bytes": size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data") / "datasets")
    parser.add_argument(
        "--out",
        type=Path,
        help="default data/kaggle/khalij-watch-chips, or khalij-watch-index with --index-only",
    )
    parser.add_argument("--dataset-id", required=True, help="<kaggle-username>/<slug>")
    parser.add_argument("--index-only", action="store_true", help="stage index and code, no shards")
    args = parser.parse_args()
    if args.out is None:
        name = "khalij-watch-index" if args.index_only else "khalij-watch-chips"
        args.out = Path("data") / "kaggle" / name

    report = stage(
        args.root,
        args.out,
        code_dirs=[ROOT / "core", ROOT / "training"],
        shards=not args.index_only,
    )
    metadata = dataset_metadata(args.dataset_id, index_only=args.index_only)
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=1), encoding="utf-8"
    )
    print(f"staged {report['shards']} shards, {report['bytes'] / 1e9:.2f} GB in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
