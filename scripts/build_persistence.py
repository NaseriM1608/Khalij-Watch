"""Build the persistent spots (core/persistence.py) from archive comparison passes.

Reads compare_cfar_model.py results, keeps passes acquired before --before
that hold the product model (picked by its weights and tiling, whatever its
label), takes its detections at the product threshold, and writes the spots
with the rule and what they were built from. The rule (core/persistence.py)
and the archive were fixed before any result (PLAN §17, 2026-10-06): S1A
passes before 2026-07-01; spots are then judged on later passes only.

Run:
    .venv-eval/Scripts/python scripts/build_persistence.py --before 2026-07-01 \
        --results models/runs/kaggle-v2/compare_val_t256 models/runs/kaggle-v2/compare_test \
                  models/runs/kaggle-v2/compare_s1a_2026h1 models/runs/kaggle-v2/compare_iran_s1a_2024-25 \
        --out data/reference/persistent_spots.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.persistence import find_spots, save_spots  # noqa: E402
from scripts.compare_cfar_model import load_result  # noqa: E402

# The product model: run 2, epoch 5, 256 px tiles, 64 px overlap, threshold 0.60.
PRODUCT_MODEL = {"sha256": "0535c82ddaa577f1", "tile": 256, "overlap": 64}
PRODUCT_THRESHOLD = 0.6


def product_peaks(result: dict) -> list | None:
    """The product model's peaks in a pass result, or None if it did not run on it."""
    for model in result["models"].values():
        if all(model.get(key) == value for key, value in PRODUCT_MODEL.items()):
            return model["peaks"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--before", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    detections, passes = [], {}
    for folder in args.results:
        for path in sorted(folder.glob("*.json")):
            if path.name == "summary.json":
                continue
            result = load_result(path)
            if dt.datetime.fromisoformat(result["acquired"]).date() >= args.before:
                continue
            peaks = product_peaks(result)
            if peaks is None:
                continue
            passes[(result["area"], result["scene"])] = True
            detections += [
                (result["area"], result["scene"], lon, lat)
                for lon, lat, score in peaks
                if score >= PRODUCT_THRESHOLD
            ]
    per_area = dict(sorted(Counter(area for area, _ in passes).items()))
    spots = find_spots(detections, per_area)
    meta = {
        "built": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "before": args.before.isoformat(),
        "results": [folder.as_posix() for folder in args.results],
        "model": PRODUCT_MODEL,
        "threshold": PRODUCT_THRESHOLD,
        "passes": per_area,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_spots(args.out, spots, meta)
    found = Counter(spot.area for spot in spots)
    for area, n in per_area.items():
        print(f"{area:15s} {n:3d} archive passes, {found[area]:3d} persistent spots", flush=True)
    print(f"{len(spots)} spots from {len(detections)} detections written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
