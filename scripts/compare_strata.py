"""Sensitivity strata of a detector comparison against GFW (compare_cfar_model.py results).

Fixed on 2026-10-05, before the Iranian-coast results were read:

- ``shore_<M>m``: only 0.01 degree cells whose centre is at least M from land
  (OSM land, core/sar/coastline.py). GFW reports no detections within about
  1 km of land, while the comparison masks only 150 m, so near-shore
  detections count as false alarms - more so in boxes with much coast. A
  detection is judged by its cell's centre, like the GFW cell it may match,
  and detections and references are left out alike.
- ``unseen_datatakes``: only passes whose datatake (satellite and datatake
  id) has no frame in the given index splits (train and val): the pair rule
  left out frames, not whole passes (scripts/list_pairs.py).

Each stratum's results are written to --work and scored by
compare_cfar_model.summarise, so the scoring is the comparison's own. Whether
GFW covered a pass (``gfw_in_box``) is never changed by a stratum. The report
goes to <work>/strata.json, never into --out (the comparison reads every
*.json there).

Run:
    .venv-eval/Scripts/python scripts/compare_strata.py \
        --out models/runs/kaggle-v2/compare_iran_s1a_2024-25 --shore-m 1000 --threshold run2=0.6
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Collection, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.areas import box_around  # noqa: E402
from core.db.spatial import BBox  # noqa: E402
from core.fusion.prefilter import distance_to_land_m  # noqa: E402
from core.sar.coastline import LandPolygon, load_land  # noqa: E402
from scripts.compare_cfar_model import (  # noqa: E402
    AREAS,
    CELL_DEG,
    cell_key,
    load_results,
    parse_threshold,
    print_summary,
    summarise,
    write_result,
)

# Land loaded this far around a box, so shore distances near its edge are exact.
LAND_MARGIN_DEG = 0.05


def datatake(scene: str) -> tuple[str, str]:
    """(satellite, datatake id) of a frame: the frames of one pass share it."""
    return scene[:3], scene.split("_")[7]


def seen_datatakes(index: Path, splits: Collection[str]) -> set[tuple[str, str]]:
    """Datatakes with a frame in these index splits."""
    seen = set()
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] in splits:
                seen.add(datatake(record["scene"]))
    return seen


def away_from_land(result: dict, land: Sequence[LandPolygon], min_m: float) -> dict:
    """A copy of a pass result keeping only cells whose centre is at least ``min_m`` from land."""
    verdicts: dict[tuple[int, int], bool] = {}

    def far(lon: float, lat: float) -> bool:
        key = cell_key(lon, lat)
        if key not in verdicts:
            row, col = key
            verdicts[key] = distance_to_land_m(col * CELL_DEG, row * CELL_DEG, land) >= min_m
        return verdicts[key]

    return {
        **result,
        "cfar": [x for x in result["cfar"] if far(x[0], x[1])],
        "models": {
            label: {**model, "peaks": [p for p in model["peaks"] if far(p[0], p[1])]}
            for label, model in result["models"].items()
        },
        "refs": [r for r in result["refs"] if far(r[0], r[1])],
    }


def land_near(box: BBox) -> list[LandPolygon]:
    m = LAND_MARGIN_DEG
    return load_land(bbox=BBox(box.min_lon - m, box.min_lat - m, box.max_lon + m, box.max_lat + m))


def stratum_summary(work: Path, name: str, results: list[dict], thresholds: dict) -> dict:
    """Write one stratum's results to its own folder and score them."""
    folder = work / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for result in results:
        write_result(
            folder / f"{result['area']}__{result['scene'].removesuffix('.SAFE')}.json", result
        )
    return summarise(folder, thresholds=thresholds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="a compare_cfar_model.py --out")
    parser.add_argument("--work", type=Path, help="default: <out>-strata")
    parser.add_argument("--shore-m", type=float, default=1000.0)
    parser.add_argument(
        "--index", type=Path, help="index for the unseen-datatake strata (omit to skip them)"
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--threshold", action="append", type=parse_threshold, default=[])
    args = parser.parse_args()
    if not args.out.is_dir() or not any(p.name != "summary.json" for p in args.out.glob("*.json")):
        parser.error(f"no comparison results in {args.out}")
    work = args.work or args.out.with_name(args.out.name + "-strata")
    thresholds = dict(args.threshold)

    results = load_results(args.out)
    land = {area: land_near(box_around(*AREAS[area])) for area in {r["area"] for r in results}}
    shore = f"shore_{args.shore_m:.0f}m"
    away = [away_from_land(r, land[r["area"]], args.shore_m) for r in results]
    strata = {"all": results, shore: away}
    if args.index:
        seen = seen_datatakes(args.index, {s.strip() for s in args.splits.split(",")})
        unseen = [datatake(r["scene"]) not in seen for r in results]
        strata["unseen_datatakes"] = [r for r, keep in zip(results, unseen, strict=True) if keep]
        strata[f"unseen_datatakes_{shore}"] = [
            r for r, keep in zip(away, unseen, strict=True) if keep
        ]
    report = {}
    for name, members in strata.items():
        report[name] = stratum_summary(work, name, members, thresholds)
        print(f"\n===== {name}: {report[name]['scenes']} passes =====", flush=True)
        print_summary(report[name])
    (work / "strata.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
