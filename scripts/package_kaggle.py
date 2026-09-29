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
    "Fishing Watch SAR vessel detections matched to AIS (non-commercial use only). Contains "
    "modified Copernicus Sentinel data. Platform positions: Offshore Platform Dataset "
    "(CC BY 4.0). Coastline: OpenStreetMap contributors (ODbL). Private research dataset."
)


def dataset_metadata(dataset_id: str) -> dict[str, Any]:
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


def stage(root: Path, out: Path, *, code_dirs: list[Path]) -> dict[str, Any]:
    """Assemble the upload folder; returns counts and total bytes."""
    out.mkdir(parents=True, exist_ok=True)
    shards: set[str] = set()
    with (root / "index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                shards.add(json.loads(line)["shard"])
    for name in ("index.jsonl", "index_meta.json"):
        shutil.copy2(root / name, out / name)
    for shard in sorted(shards):
        _link_or_copy(root / shard, out / shard)
    for code in code_dirs:
        target = out / "code" / code.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(code, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {"shards": len(shards), "bytes": size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data") / "datasets")
    parser.add_argument("--out", type=Path, default=Path("data") / "kaggle" / "khalij-watch-chips")
    parser.add_argument("--dataset-id", required=True, help="<kaggle-username>/<slug>")
    args = parser.parse_args()

    report = stage(args.root, args.out, code_dirs=[ROOT / "core", ROOT / "training"])
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata(args.dataset_id), indent=1), encoding="utf-8"
    )
    print(f"staged {report['shards']} shards, {report['bytes'] / 1e9:.2f} GB in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
