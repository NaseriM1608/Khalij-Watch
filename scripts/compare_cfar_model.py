"""Compare CFAR and the trained point detector on held-out passes, against GFW.

Our labels are CFAR detections that GFW matched to AIS, so CFAR cannot be
scored fairly against them. The independent reference here is every GFW SAR
vessel detection (AIS-matched and unmatched) in the box at the pass hour.
GFW publishes detections as counts per 0.01 degree cell, so both detectors
are scored per cell: a detection matches while its cell (or, in the lenient
score, a neighbouring cell) still has unmatched GFW detections. Both run on
the same window with the same land, coast-buffer and platform mask, and
reference cells whose centre is masked are left out for both.

One JSON per (area, scene) under --out, so a rerun resumes; --summary only
aggregates what exists.

Run:
    .venv-train/Scripts/python scripts/compare_cfar_model.py --model models/runs/kaggle-v1/local_best.onnx
    .venv-train/Scripts/python scripts/compare_cfar_model.py --summary
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from core.inference.runner import run_tiled  # noqa: E402
from core.labeling.gfw import PASS_MARGIN  # noqa: E402

CELL_DEG = 0.01
MODEL_STORE_THRESHOLD = 0.05  # peaks kept for threshold sweeps
MODEL_THRESHOLD = 0.45  # chosen on validation (kaggle-v1, epoch 6)
# 128 px tiles were chosen for run 1, which trained on 128 px chips; run 2
# trained on random 96 px crops of them and a tile-size check for it is
# still pending. Small tiles matter because EfficientNet's squeeze-excitation
# blocks average over the whole input: on 2048 px tiles of calm, dark sea the
# model went blind (0 peaks where 128 px tiles find 128; 2024-05-16 Fujairah).
MODEL_TILE = 128
MODEL_OVERLAP = 32
AREAS = {
    "fujairah": (56.53, 25.22, 20.0),
    "khor_fakkan": (56.52, 25.41, 20.0),
    "sharjah_dubai": (55.33, 25.47, 20.0),
    "ras_al_khaimah": (55.88, 26.02, 20.0),
    "hormuz_lanes": (56.55, 26.50, 40.0),
}
OUT = Path("models/runs/kaggle-v1/compare")


def cell_key(lon: float, lat: float) -> tuple[int, int]:
    """GFW 4Wings cell of a position (cells are centred on 0.01 degree multiples)."""
    return round(lat / CELL_DEG), round(lon / CELL_DEG)


def match_cells(
    preds: Sequence[tuple[float, float]],
    refs: Sequence[tuple[float, float, int]],
    *,
    neighbours: bool,
) -> tuple[Counter, int]:
    """Reference detections hit per cell, and the number of unmatched detections.

    Same-cell matches are taken first; the neighbour pass is greedy (on the
    kaggle-v1 comparison an optimal matching adds 1 hit for CFAR and 3 for the
    model out of ~1,600).
    """
    remaining = Counter()
    for lon, lat, count in refs:
        remaining[cell_key(lon, lat)] += count
    unmatched = Counter(cell_key(lon, lat) for lon, lat in preds)
    hits = Counter()
    for key in list(unmatched):
        take = min(unmatched[key], remaining[key])
        hits[key] += take
        unmatched[key] -= take
        remaining[key] -= take
    if neighbours:
        for (row, col), left in list(unmatched.items()):
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    other = (row + d_row, col + d_col)
                    take = min(left, remaining[other])
                    hits[other] += take
                    left -= take
                    remaining[other] -= take
            unmatched[(row, col)] = left
    return hits, sum(unmatched.values())


def score_cells(
    preds: Sequence[tuple[float, float]],
    refs: Sequence[tuple[float, float, int]],
    *,
    neighbours: bool,
) -> tuple[int, int, int]:
    """(tp, fp, fn) of detections against per-cell reference counts."""
    hits, fp = match_cells(preds, refs, neighbours=neighbours)
    tp = sum(hits.values())
    return tp, fp, sum(count for _, _, count in refs) - tp


def test_pairs(index: Path) -> list[tuple[str, str]]:
    pairs = set()
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            if '"test"' in line:
                record = json.loads(line)
                if record["split"] == "test":
                    pairs.add((record["area"], record["scene"]))
    return sorted(pairs)


def spread(pairs: list[tuple[str, str]], per_area: int) -> list[tuple[str, str]]:
    """Up to ``per_area`` passes per area, spread evenly through its scenes in name order.

    Scene names sort by acquisition time only within one satellite (all test
    passes so far are S1A). With more than ``per_area`` scenes, index
    ``int(i * len / per_area)`` is taken for each i, so the first scene is
    always chosen and the last one or two may never be. Chosen before any
    result; kept as is so later runs are compared on the same passes.
    """
    chosen = []
    for area in sorted({a for a, _ in pairs}):
        scenes = sorted(s for a, s in pairs if a == area)
        if len(scenes) <= per_area:
            chosen += [(area, s) for s in scenes]
            continue
        step = len(scenes) / per_area
        chosen += [(area, scenes[int(i * step)]) for i in range(per_area)]
    return chosen


def gfw_days(acquired: dt.datetime) -> tuple[dt.date, dt.date]:
    """UTC days [start, end) to fetch so every hour cells_for_pass keeps is included."""
    first = (acquired - PASS_MARGIN).astimezone(dt.UTC).date()
    last = (acquired + PASS_MARGIN).astimezone(dt.UTC).date()
    return first, last + dt.timedelta(days=1)


def unmasked(masked, row: float, col: float) -> bool:
    """Whether a window-relative (row, col) rounds to an unmasked pixel inside the window.

    The one test applied to CFAR candidates, model peaks and reference cells.
    """
    r, c = round(row), round(col)
    return 0 <= r < masked.shape[0] and 0 <= c < masked.shape[1] and not masked[r, c]


def reference_cells(cells, box, scene, offset, masked) -> tuple[list, int]:
    """GFW cells in the box as (lon, lat, detections, matched), and the detections masked out.

    ``scene.raster_coords`` is corner-based, so a cell centre is moved back half
    a pixel before it is looked up in the window.
    """
    refs, dropped = [], 0
    for cell in cells:
        if not box.contains(cell.lon, cell.lat):
            continue
        rows, cols = scene.raster_coords([cell.lon], [cell.lat])
        if not unmasked(masked, rows[0] - 0.5 - offset[0], cols[0] - 0.5 - offset[1]):
            dropped += cell.detections
            continue
        refs.append((cell.lon, cell.lat, cell.detections, cell.matched))
    return refs, dropped


def compare_scene(area: str, scene_name: str, session, gfw_client) -> dict:
    """CFAR detections, model peaks and GFW reference cells for one test pass."""
    from core.inference.contract import normalise_chip
    from core.inference.decode import decode_peaks
    from core.labeling.gfw import cells_for_pass
    from core.labeling.gfw_api import fetch_sar_cells
    from core.sar.cfar import CfarConfig
    from core.sar.coastline import DEFAULT_BUFFER_M
    from core.sar.detect import detect_candidates, georeference
    from scripts.build_dataset import exclusion_mask, passes_over, read_pair
    from scripts.detect_window import box_around

    lon, lat, size = AREAS[area]
    box = box_around(lon, lat, size)
    day = dt.datetime.strptime(scene_name.split("_")[4][:8], "%Y%m%d").replace(tzinfo=dt.UTC)
    product = next(
        p
        for p in passes_over(box, day - dt.timedelta(days=1), day + dt.timedelta(days=2))
        if p.name == scene_name
    )
    started = time.perf_counter()
    scene, window, bands = read_pair(product, box)
    masked = exclusion_mask(box, product, scene, window, bands, DEFAULT_BUFFER_M)
    offset = (window.row_off, window.col_off)

    config = CfarConfig(pfa=1e-6, satellite=product.satellite)
    candidates = [
        c
        for c in detect_candidates(bands[1], config=config, mask=masked)
        if unmasked(masked, c.row, c.col)
    ]
    cfar = [
        (c.lon, c.lat)
        for c in georeference(candidates, scene, offset)
        if box.contains(c.lon, c.lat)
    ]

    def predict(x):
        return tuple(session.run(["objectness", "offset"], {"image": x}))

    heat, off = run_tiled(predict, normalise_chip(bands), tile=MODEL_TILE, overlap=MODEL_OVERLAP)
    peaks = [
        (r, c, s)
        for r, c, s in decode_peaks(heat, off, threshold=MODEL_STORE_THRESHOLD)
        if unmasked(masked, r, c)
    ]
    model = []
    if peaks:
        lons, lats = scene.lonlat(
            [r + offset[0] for r, _, _ in peaks], [c + offset[1] for _, c, _ in peaks]
        )
        model = [
            (float(x), float(y), s)
            for x, y, (_, _, s) in zip(lons, lats, peaks, strict=True)
            if box.contains(float(x), float(y))
        ]

    cells = asyncio.run(fetch_sar_cells(gfw_client, box, *gfw_days(product.acq_start)))
    refs, dropped = reference_cells(
        cells_for_pass(cells, product.acq_start), box, scene, offset, masked
    )
    return {
        "area": area,
        "scene": scene_name,
        "acquired": product.acq_start.isoformat(),
        "cfar": cfar,
        "model": model,
        "refs": refs,
        "refs_dropped_masked": dropped,
        "model_tile": MODEL_TILE,
        "seconds": round(time.perf_counter() - started, 1),
    }


def split_hits(hits: Counter, refs: Sequence[tuple[float, float, int, bool]]) -> dict:
    """Hits per AIS flag, from one joint assignment.

    GFW gives counts per cell, not identities, so in a cell holding both
    AIS-matched and unmatched detections a cell's hits are shared in proportion
    to their counts; ``unmatched_min``/``_max`` bound what any choice would give.
    """
    per_cell: dict = {}
    for lon, lat, count, matched in refs:
        counts = per_cell.setdefault(cell_key(lon, lat), [0, 0])
        counts[0 if matched else 1] += count
    split = {"matched": 0.0, "unmatched": 0.0, "unmatched_min": 0, "unmatched_max": 0}
    for key, (n_matched, n_unmatched) in per_cell.items():
        hit = hits[key]
        if hit:
            split["matched"] += hit * n_matched / (n_matched + n_unmatched)
            split["unmatched"] += hit * n_unmatched / (n_matched + n_unmatched)
            split["unmatched_min"] += max(0, hit - n_matched)
            split["unmatched_max"] += min(hit, n_unmatched)
    return split


def summarise(out: Path, threshold: float = MODEL_THRESHOLD) -> dict:
    """Precision, recall and F1 per detector, area and matching rule; recall on matched/unmatched."""
    results = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(out.glob("*.json"))
        if p.name != "summary.json"
    ]
    table: dict = {}
    for neighbours in (False, True):
        rule = "neighbour_cells" if neighbours else "same_cell"
        for area in ["all", *sorted({r["area"] for r in results})]:
            rows = [r for r in results if area in ("all", r["area"])]
            for name in ("cfar", "model"):
                tp = fp = n_m = n_u = 0
                tp_m = tp_u = 0.0
                tp_u_min = tp_u_max = 0
                for r in rows:
                    preds = [(x[0], x[1]) for x in r[name] if name == "cfar" or x[2] >= threshold]
                    refs = [(x[0], x[1], x[2], x[3]) for x in r["refs"]]
                    hits, wrong = match_cells(preds, [x[:3] for x in refs], neighbours=neighbours)
                    split = split_hits(hits, refs)
                    tp, fp = tp + sum(hits.values()), fp + wrong
                    tp_m, tp_u = tp_m + split["matched"], tp_u + split["unmatched"]
                    tp_u_min += split["unmatched_min"]
                    tp_u_max += split["unmatched_max"]
                    n_m += sum(x[2] for x in refs if x[3])
                    n_u += sum(x[2] for x in refs if not x[3])
                fn = n_m + n_u - tp
                p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                table.setdefault(rule, {}).setdefault(area, {})[name] = {
                    "precision": p,
                    "recall": rc,
                    "f1": 2 * p * rc / max(p + rc, 1e-9),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tp_ais_matched": tp_m,
                    "tp_unmatched": tp_u,
                    "recall_ais_matched": tp_m / max(n_m, 1),
                    "recall_unmatched": tp_u / max(n_u, 1),
                    "recall_unmatched_range": [tp_u_min / max(n_u, 1), tp_u_max / max(n_u, 1)],
                    "ref_matched": n_m,
                    "ref_unmatched": n_u,
                }
    return {"scenes": len(results), "model_threshold": threshold, "table": table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/runs/kaggle-v1/local_best.onnx"))
    parser.add_argument("--index", type=Path, default=Path("data/datasets/index.jsonl"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--per-area", type=int, default=0, help="evenly spaced passes per area (0 = all)"
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if not args.summary:
        import gfwapiclient
        import onnxruntime as ort

        from core.config import load_settings
        from core.sar.reader import S3AccessError

        load_settings()
        session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        client = gfwapiclient.Client(access_token=os.environ["GFW_API_TOKEN"])
        args.out.mkdir(parents=True, exist_ok=True)
        pairs = test_pairs(args.index)
        if args.per_area:
            pairs = spread(pairs, args.per_area)
        todo = [
            (a, s)
            for a, s in pairs
            if not (args.out / f"{a}__{s.removesuffix('.SAFE')}.json").exists()
        ]
        if args.limit:
            todo = todo[: args.limit]
        print(f"{len(todo)} test passes to compare", flush=True)
        failed = 0
        for area, scene_name in todo:
            try:
                result = compare_scene(area, scene_name, session, client)
            except (S3AccessError, ValueError, StopIteration, OSError) as exc:
                failed += 1
                print(f"  {area} {scene_name[:40]} FAILED {str(exc)[:120]}", flush=True)
                continue
            path = args.out / f"{area}__{scene_name.removesuffix('.SAFE')}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result), encoding="utf-8")
            tmp.replace(path)
            print(
                f"  {area:15s} {scene_name[17:32]} cfar {len(result['cfar']):3d} "
                f"model {sum(m[2] >= MODEL_THRESHOLD for m in result['model']):3d} "
                f"gfw {sum(r[2] for r in result['refs']):3d} {result['seconds']}s",
                flush=True,
            )
        if failed:
            print(f"{failed} passes failed; rerun to retry", flush=True)
    if not args.out.is_dir() or not any(p.name != "summary.json" for p in args.out.glob("*.json")):
        print(f"no results in {args.out}; nothing to summarise", flush=True)
        return 1
    summary = summarise(args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for rule, areas in summary["table"].items():
        print(
            f"\n{rule} ({summary['scenes']} passes, model threshold {summary['model_threshold']})"
        )
        for area, both in areas.items():
            for name, m in both.items():
                low, high = m["recall_unmatched_range"]
                print(
                    f"  {area:15s} {name:5s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
                    f"| recall AIS-matched {m['recall_ais_matched']:.3f} unmatched {m['recall_unmatched']:.3f} "
                    f"({low:.3f}-{high:.3f}, n {m['ref_unmatched']})"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
