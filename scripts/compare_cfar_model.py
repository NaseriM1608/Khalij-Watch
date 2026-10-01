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

CELL_DEG = 0.01
MODEL_STORE_THRESHOLD = 0.05  # peaks kept for threshold sweeps
MODEL_THRESHOLD = 0.45  # chosen on validation (kaggle-v1, epoch 6)
# Inference tiles at the training chip size. EfficientNet's squeeze-excitation
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


def score_cells(
    preds: Sequence[tuple[float, float]],
    refs: Sequence[tuple[float, float, int]],
    *,
    neighbours: bool,
) -> tuple[int, int, int]:
    """(tp, fp, fn) of detections against per-cell reference counts."""
    remaining = Counter()
    for lon, lat, count in refs:
        remaining[cell_key(lon, lat)] += count
    unmatched = Counter(cell_key(lon, lat) for lon, lat in preds)
    tp = 0
    for key in list(unmatched):
        take = min(unmatched[key], remaining[key])
        tp += take
        unmatched[key] -= take
        remaining[key] -= take
    if neighbours:
        for (row, col), left in list(unmatched.items()):
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    other = (row + d_row, col + d_col)
                    take = min(left, remaining[other])
                    tp += take
                    left -= take
                    remaining[other] -= take
            unmatched[(row, col)] = left
    return tp, sum(unmatched.values()), sum(remaining.values())


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
    """Up to ``per_area`` passes per area, evenly spaced in time (chosen before any result)."""
    chosen = []
    for area in sorted({a for a, _ in pairs}):
        scenes = sorted(s for a, s in pairs if a == area)
        if len(scenes) <= per_area:
            chosen += [(area, s) for s in scenes]
            continue
        step = len(scenes) / per_area
        chosen += [(area, scenes[int(i * step)]) for i in range(per_area)]
    return chosen


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
    cfar = [
        (c.lon, c.lat)
        for c in georeference(
            detect_candidates(bands[1], config=config, mask=masked), scene, offset
        )
        if box.contains(c.lon, c.lat)
    ]

    def predict(x):
        return tuple(session.run(["objectness", "offset"], {"image": x}))

    heat, off = run_tiled(predict, normalise_chip(bands), tile=MODEL_TILE, overlap=MODEL_OVERLAP)
    peaks = [
        (r, c, s)
        for r, c, s in decode_peaks(heat, off, threshold=MODEL_STORE_THRESHOLD)
        if 0 <= round(r) < masked.shape[0]
        and 0 <= round(c) < masked.shape[1]
        and not masked[round(r), round(c)]
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

    cells = asyncio.run(
        fetch_sar_cells(gfw_client, box, day.date(), (day + dt.timedelta(days=1)).date())
    )
    refs, dropped = [], 0
    for cell in cells_for_pass(cells, product.acq_start):
        if not box.contains(cell.lon, cell.lat):
            continue
        rows, cols = scene.raster_coords([cell.lon], [cell.lat])
        r, c = round(rows[0] - 0.5 - offset[0]), round(cols[0] - 0.5 - offset[1])
        if not (0 <= r < masked.shape[0] and 0 <= c < masked.shape[1]) or masked[r, c]:
            dropped += cell.detections
            continue
        refs.append((cell.lon, cell.lat, cell.detections, cell.matched))
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
                tp = fp = fn = tp_m = n_m = tp_u = n_u = 0
                for r in rows:
                    preds = [(x[0], x[1]) for x in r[name] if name == "cfar" or x[2] >= threshold]
                    refs = [(x[0], x[1], x[2]) for x in r["refs"]]
                    a, b, c = score_cells(preds, refs, neighbours=neighbours)
                    tp, fp, fn = tp + a, fp + b, fn + c
                    for flag in (True, False):
                        sub = [(x[0], x[1], x[2]) for x in r["refs"] if x[3] is flag]
                        hit, _, _ = score_cells(preds, sub, neighbours=neighbours)
                        total = sum(x[2] for x in sub)
                        if flag:
                            tp_m, n_m = tp_m + hit, n_m + total
                        else:
                            tp_u, n_u = tp_u + hit, n_u + total
                p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                table.setdefault(rule, {}).setdefault(area, {})[name] = {
                    "precision": p,
                    "recall": rc,
                    "f1": 2 * p * rc / max(p + rc, 1e-9),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "recall_ais_matched": tp_m / max(n_m, 1),
                    "recall_unmatched": tp_u / max(n_u, 1),
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
    summary = summarise(args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for rule, areas in summary["table"].items():
        print(
            f"\n{rule} ({summary['scenes']} passes, model threshold {summary['model_threshold']})"
        )
        for area, both in areas.items():
            for name, m in both.items():
                print(
                    f"  {area:15s} {name:5s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
                    f"| recall AIS-matched {m['recall_ais_matched']:.3f} unmatched {m['recall_unmatched']:.3f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
