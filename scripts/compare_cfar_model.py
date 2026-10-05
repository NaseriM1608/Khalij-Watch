"""Compare CFAR and trained point detectors on held-out passes, against GFW.

Our labels are CFAR detections that GFW matched to AIS, so CFAR cannot be
scored fairly against them. The independent reference here is every GFW SAR
vessel detection (AIS-matched and unmatched) in the box at the pass hour.
GFW publishes detections as counts per 0.01 degree cell, so all detectors
are scored per cell: a detection matches while its cell (or, in the lenient
score, a neighbouring cell) still has unmatched GFW detections. All run on
the same window with the same land, coast-buffer and platform mask, and
reference cells whose centre is masked are left out for all.

Each pass is read once and every --model (LABEL=PATH[@TILE/OVERLAP]) runs on
it. One JSON per (area, scene) under --out records each model's weights
(sha256) and tiling, so a rerun resumes only passes compared with the same
models, and a summary refuses a folder that mixes runs. The summary adds a
threshold sweep per model and a paired bootstrap over passes of model minus
CFAR. Thresholds and checkpoints are chosen on --split val, never on test.

Run:
    .venv-eval/Scripts/python scripts/compare_cfar_model.py --split val --per-area 5 \
        --out models/runs/kaggle-v2/compare_val \
        --model v1=models/runs/kaggle-v1/local_best.onnx \
        --model e14=models/runs/kaggle-v2/run/epoch_14.onnx@256/64
    .venv-eval/Scripts/python scripts/compare_cfar_model.py --summary --out models/runs/kaggle-v1/compare
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import string
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from core.areas import STUDY_AREAS  # noqa: E402
from core.inference.runner import check_tiling, run_tiled  # noqa: E402
from core.labeling.gfw import PASS_MARGIN  # noqa: E402

CELL_DEG = 0.01
MODEL_STORE_THRESHOLD = 0.05  # peaks kept for threshold sweeps
MODEL_THRESHOLD = 0.45  # chosen on validation (kaggle-v1, epoch 6)
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(1, 19))  # 0.05 .. 0.90
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
# 128 px tiles were chosen for run 1, which trained on 128 px chips; run 2
# trained on random 96 px crops of them and a tile-size check for it is
# still pending. Small tiles matter because EfficientNet's squeeze-excitation
# blocks average over the whole input: on 2048 px tiles of calm, dark sea the
# model went blind (0 peaks where 128 px tiles find 128; 2024-05-16 Fujairah).
MODEL_TILE = 128
MODEL_OVERLAP = 32
DEFAULT_MODEL = Path("models/runs/kaggle-v1/local_best.onnx")
LABEL_CHARS = set(string.ascii_letters + string.digits + "_-.")
AREAS = STUDY_AREAS
OUT = Path("models/runs/kaggle-v1/compare")
# Per-pass counts, in this order: true and false positives, GFW detections with
# and without AIS, hits on each (shared by count in mixed cells), and the
# least and most unmatched hits any sharing could give.
COUNT_FIELDS = (
    "tp",
    "fp",
    "n_matched",
    "n_unmatched",
    "tp_matched",
    "tp_unmatched",
    "tp_unmatched_min",
    "tp_unmatched_max",
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    tile: int = MODEL_TILE
    overlap: int = MODEL_OVERLAP


def parse_model_spec(text: str) -> ModelSpec:
    """LABEL=PATH[@TILE/OVERLAP]; a bare PATH is labelled "model"."""
    label, sep, rest = text.partition("=")
    if not sep:
        label, rest = "model", text
    if not label or label == "cfar" or not set(label) <= LABEL_CHARS:
        raise ValueError(f"bad model label {label!r} in {text!r}")
    path, at, tiling = rest.partition("@")
    if not at:
        return ModelSpec(label, Path(path))
    tile, slash, overlap = tiling.partition("/")
    if not slash:
        raise ValueError(f"tiling must be TILE/OVERLAP in {text!r}")
    spec = ModelSpec(label, Path(path), int(tile), int(overlap))
    check_tiling(spec.tile, spec.overlap)
    return spec


def parse_threshold(text: str) -> tuple[str, float]:
    """LABEL=VALUE."""
    label, sep, value = text.partition("=")
    if not sep or not label:
        raise ValueError(f"threshold must be LABEL=VALUE, not {text!r}")
    return label, float(value)


def provenance(spec: ModelSpec) -> dict:
    """What decides a model's peaks: its weights (sha256) and tiling; the path is for people."""
    digest = hashlib.sha256(spec.path.read_bytes()).hexdigest()[:16]
    return {
        "path": spec.path.as_posix(),
        "sha256": digest,
        "tile": spec.tile,
        "overlap": spec.overlap,
    }


def model_key(model: dict) -> tuple:
    return model.get("sha256"), model["tile"], model["overlap"]


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


def split_pairs(index: Path, split: str) -> list[tuple[str, str]]:
    """Sorted distinct (area, scene) pairs of one split."""
    pairs = set()
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            if f'"{split}"' in line:
                record = json.loads(line)
                if record["split"] == split:
                    pairs.add((record["area"], record["scene"]))
    return sorted(pairs)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    """(area, scene) pairs from a text file, one "area scene" per line; # starts a comment.

    For passes outside the dataset index, such as Sentinel-1C/1D passes after
    GFW's recalibration in September 2026.
    """
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        area, scene = line.split()
        if area not in AREAS:
            raise ValueError(f"unknown area {area!r} in {path}")
        pairs.append((area, scene))
    return pairs


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


def sea_levels(bands, masked) -> list[float]:
    """Median VV and VH amplitude (DN) of valid, unmasked pixels: mostly sea.

    The model's input normalisation was frozen on Sentinel-1A; if another
    satellite's calibration differs, this level moves.
    """
    keep = ~masked & (bands > 0).all(axis=0)
    return [float(np.median(band[keep])) if keep.any() else 0.0 for band in bands]


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


def model_peaks(
    session, image, masked, scene, offset, box, *, tile: int, overlap: int
) -> list[tuple[float, float, float]]:
    """(lon, lat, score) of a model's peaks above MODEL_STORE_THRESHOLD, unmasked and in the box."""
    from core.inference.decode import decode_peaks

    def predict(x):
        return tuple(session.run(["objectness", "offset"], {"image": x}))

    heat, off = run_tiled(predict, image, tile=tile, overlap=overlap)
    peaks = [
        (r, c, s)
        for r, c, s in decode_peaks(heat, off, threshold=MODEL_STORE_THRESHOLD)
        if unmasked(masked, r, c)
    ]
    if not peaks:
        return []
    lons, lats = scene.lonlat(
        [r + offset[0] for r, _, _ in peaks], [c + offset[1] for _, c, _ in peaks]
    )
    return [
        (float(x), float(y), s)
        for x, y, (_, _, s) in zip(lons, lats, peaks, strict=True)
        if box.contains(float(x), float(y))
    ]


def compare_scene(area: str, scene_name: str, loaded, gfw_client) -> dict:
    """CFAR detections, every model's peaks and GFW reference cells for one pass.

    ``loaded`` holds (ModelSpec, provenance, onnxruntime session) per model.
    """
    from core.inference.contract import normalise_chip
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
    # GFW first: when it fails, the pass fails before the image read and every model run.
    cells = asyncio.run(fetch_sar_cells(gfw_client, box, *gfw_days(product.acq_start)))
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

    image = normalise_chip(bands)
    models = {}
    for spec, prov, session in loaded:
        began = time.perf_counter()
        peaks = model_peaks(
            session, image, masked, scene, offset, box, tile=spec.tile, overlap=spec.overlap
        )
        models[spec.label] = {
            **prov,
            "peaks": peaks,
            "seconds": round(time.perf_counter() - began, 1),
        }

    refs, dropped = reference_cells(
        cells_for_pass(cells, product.acq_start), box, scene, offset, masked
    )
    return {
        "area": area,
        "scene": scene_name,
        "acquired": product.acq_start.isoformat(),
        "cfar": cfar,
        "models": models,
        "refs": refs,
        "refs_dropped_masked": dropped,
        # GFW detections in the box before masking: none means GFW did not cover the pass.
        "gfw_in_box": sum(ref[2] for ref in refs) + dropped,
        "sea_dn": sea_levels(bands, masked),
        "seconds": round(time.perf_counter() - started, 1),
    }


def result_path(out: Path, area: str, scene_name: str) -> Path:
    return out / f"{area}__{scene_name.removesuffix('.SAFE')}.json"


def load_result(path: Path) -> dict:
    """A pass result; one written before several models per pass (run 1) holds one, "model"."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if "models" not in result:
        result["models"] = {
            "model": {
                "path": None,
                "sha256": None,
                "tile": result.get("model_tile", MODEL_TILE),
                "overlap": MODEL_OVERLAP,
                "peaks": result.get("model", []),
            }
        }
    return result


def is_done(path: Path, wanted: dict[str, dict]) -> bool:
    """Whether a pass was compared with exactly these models, same weights and tiling."""
    if not path.exists():
        return False
    try:
        models = load_result(path)["models"]
        keys = {label: model_key(m) for label, m in models.items()}
    except (ValueError, KeyError, TypeError):  # cut short by a power cut: compare again
        return False
    return set(keys) == set(wanted) and all(
        keys[label] == model_key(prov) for label, prov in wanted.items()
    )


def write_result(path: Path, result: dict) -> None:
    """Through a flushed temporary file, so a power cut leaves the old file or the new one."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(result, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def load_results(out: Path) -> list[dict]:
    """Every pass result under ``out``; all must hold the same models, or a table would mix runs."""
    results = [load_result(p) for p in sorted(out.glob("*.json")) if p.name != "summary.json"]
    keys = [{label: model_key(m) for label, m in r["models"].items()} for r in results]
    for result, key in zip(results, keys, strict=True):
        if key != keys[0]:
            raise ValueError(
                f"{result['area']} {result['scene']} holds other models than "
                f"{results[0]['scene']}; summarise one run per --out"
            )
    return results


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


def predictions(result: dict, name: str, threshold: float | None) -> list[tuple[float, float]]:
    if name == "cfar":
        return [(x[0], x[1]) for x in result["cfar"]]
    return [(x[0], x[1]) for x in result["models"][name]["peaks"] if x[2] >= threshold]


def pass_counts(result: dict, preds, *, neighbours: bool) -> list[float]:
    """One pass's counts, in COUNT_FIELDS order, from one joint assignment."""
    refs = [(x[0], x[1], x[2], x[3]) for x in result["refs"]]
    hits, wrong = match_cells(preds, [x[:3] for x in refs], neighbours=neighbours)
    split = split_hits(hits, refs)
    return [
        sum(hits.values()),
        wrong,
        sum(x[2] for x in refs if x[3]),
        sum(x[2] for x in refs if not x[3]),
        split["matched"],
        split["unmatched"],
        split["unmatched_min"],
        split["unmatched_max"],
    ]


def rates(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and subset recalls of summed counts (..., COUNT_FIELDS)."""
    tp, fp, n_m, n_u, tp_m, tp_u = (counts[..., i] for i in range(6))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(n_m + n_u, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / np.maximum(precision + recall, 1e-9),
        "recall_ais_matched": tp_m / np.maximum(n_m, 1),
        "recall_unmatched": tp_u / np.maximum(n_u, 1),
    }


def metrics(counts: np.ndarray) -> dict:
    """The summary row of one detector from its summed counts."""
    tp, fp, n_m, n_u, tp_m, tp_u, tp_u_min, tp_u_max = (float(x) for x in counts)
    return {
        **{key: float(value) for key, value in rates(np.asarray(counts)).items()},
        "tp": round(tp),
        "fp": round(fp),
        "fn": round(n_m + n_u - tp),
        "tp_ais_matched": tp_m,
        "tp_unmatched": tp_u,
        "recall_unmatched_range": [tp_u_min / max(n_u, 1), tp_u_max / max(n_u, 1)],
        "ref_matched": round(n_m),
        "ref_unmatched": round(n_u),
    }


def all_counts(results: list[dict], name: str, threshold, *, neighbours: bool) -> np.ndarray:
    """(passes, COUNT_FIELDS) counts of one detector."""
    rows = [pass_counts(r, predictions(r, name, threshold), neighbours=neighbours) for r in results]
    return np.array(rows, dtype=float).reshape(len(results), len(COUNT_FIELDS))


def sweep(results: list[dict], label: str, *, neighbours: bool) -> dict:
    """F1, precision, recall and unmatched recall at each threshold; the lowest best one."""
    by_threshold = {}
    for threshold in THRESHOLD_GRID:
        row = metrics(all_counts(results, label, threshold, neighbours=neighbours).sum(0))
        by_threshold[f"{threshold:.2f}"] = {
            key: row[key] for key in ("f1", "precision", "recall", "recall_unmatched")
        }
    best = max(THRESHOLD_GRID, key=lambda t: by_threshold[f"{t:.2f}"]["f1"])
    return {
        "best_threshold": best,
        "best_f1": by_threshold[f"{best:.2f}"]["f1"],
        "by_threshold": by_threshold,
    }


def paired_bootstrap(cfar: np.ndarray, model: np.ndarray, resamples: int) -> dict:
    """Model minus CFAR for F1 and subset recalls, with 95% intervals from resampling passes.

    Both detectors ran on the same passes, so each resample draws whole passes
    for both: the interval reflects which passes happened to be compared.
    """
    if len(cfar) == 0 or resamples <= 0:
        return {}
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(cfar), size=(resamples, len(cfar))
    )
    full_c, full_m = rates(cfar.sum(0)), rates(model.sum(0))
    boot_c, boot_m = rates(cfar[draws].sum(1)), rates(model[draws].sum(1))
    out: dict = {"resamples": resamples, "seed": BOOTSTRAP_SEED}
    for key in ("f1", "recall_ais_matched", "recall_unmatched"):
        out[key] = {
            "diff": float(full_m[key] - full_c[key]),
            "ci95": [float(v) for v in np.percentile(boot_m[key] - boot_c[key], [2.5, 97.5])],
            "cfar_ci95": [float(v) for v in np.percentile(boot_c[key], [2.5, 97.5])],
            "model_ci95": [float(v) for v in np.percentile(boot_m[key], [2.5, 97.5])],
        }
    return out


def summarise(
    out: Path,
    threshold: float = MODEL_THRESHOLD,
    thresholds: dict[str, float] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Per detector, area and matching rule: P, R, F1 and subset recalls; per model a threshold
    sweep and a paired bootstrap against CFAR."""
    results = load_results(out)
    # A pass with no GFW detection anywhere in the box is one GFW did not cover
    # (no Sentinel-1C data in 2025, offline July-September 2026), not an empty
    # sea: scoring it would turn every detection into a false alarm.
    excluded = [r["scene"] for r in results if r.get("gfw_in_box", 1) == 0]
    results = [r for r in results if r.get("gfw_in_box", 1) != 0]
    labels = list(results[0]["models"]) if results else ["model"]
    unknown = set(thresholds or {}) - set(labels)
    if unknown:
        raise ValueError(f"thresholds for models not in {out}: {', '.join(sorted(unknown))}")
    chosen = {label: (thresholds or {}).get(label, threshold) for label in labels}
    areas = ["all", *sorted({r["area"] for r in results})]
    table: dict = {}
    sweeps: dict = {label: {} for label in labels}
    bootstrap: dict = {label: {} for label in labels}
    for neighbours in (False, True):
        rule = "neighbour_cells" if neighbours else "same_cell"
        counts = {
            name: all_counts(results, name, chosen.get(name), neighbours=neighbours)
            for name in ("cfar", *labels)
        }
        for area in areas:
            rows = [i for i, r in enumerate(results) if area in ("all", r["area"])]
            table.setdefault(rule, {})[area] = {
                name: metrics(c[rows].sum(0)) for name, c in counts.items()
            }
        for label in labels:
            sweeps[label][rule] = sweep(results, label, neighbours=neighbours)
            boot = paired_bootstrap(counts["cfar"], counts[label], resamples)
            bootstrap[label][rule] = {"threshold": chosen[label], **boot} if boot else {}
    return {
        "scenes": len(results),
        "excluded_no_gfw": excluded,
        "splits": sorted({r.get("split", "unrecorded") for r in results}),
        "model_threshold": threshold,
        "thresholds": chosen,
        "models": {
            label: {
                **{
                    k: v
                    for k, v in results[0]["models"][label].items()
                    if k not in ("peaks", "seconds")
                },
                "seconds_total": round(
                    sum(r["models"][label].get("seconds", 0) for r in results), 1
                ),
            }
            for label in (labels if results else [])
        },
        "table": table,
        "sweep": sweeps,
        "bootstrap": bootstrap,
    }


def run_passes(args: argparse.Namespace, specs: list[ModelSpec], thresholds: dict) -> None:
    """Compare every pass of the split not yet compared with exactly these models."""
    import gfwapiclient
    import onnxruntime as ort
    from gfwapiclient.exceptions import GFWAPIClientError

    from core.config import load_settings
    from core.sar.catalogue import CatalogueError
    from core.sar.reader import S3AccessError

    load_settings()
    loaded = [
        (
            spec,
            provenance(spec),
            ort.InferenceSession(str(spec.path), providers=["CPUExecutionProvider"]),
        )
        for spec in specs
    ]
    wanted = {spec.label: prov for spec, prov, _ in loaded}
    client = gfwapiclient.Client(access_token=os.environ["GFW_API_TOKEN"])
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pairs:
        pairs, split = read_pairs(args.pairs), args.pairs.stem
    else:
        pairs, split = split_pairs(args.index, args.split), args.split
    if args.per_area:
        pairs = spread(pairs, args.per_area)
    todo = [(a, s) for a, s in pairs if not is_done(result_path(args.out, a, s), wanted)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} {split} passes to compare with {', '.join(wanted)}", flush=True)
    failed = 0
    for area, scene_name in todo:
        try:
            result = compare_scene(area, scene_name, loaded, client)
        except (
            S3AccessError,
            CatalogueError,
            GFWAPIClientError,
            ValueError,
            StopIteration,
            OSError,
        ) as exc:
            failed += 1
            print(
                f"  {area} {scene_name[:40]} FAILED {type(exc).__name__}: {str(exc)[:120]}",
                flush=True,
            )
            continue
        result["split"] = split
        write_result(result_path(args.out, area, scene_name), result)
        found = " ".join(
            f"{label} {sum(p[2] >= thresholds.get(label, MODEL_THRESHOLD) for p in m['peaks'])}"
            for label, m in result["models"].items()
        )
        print(
            f"  {area:15s} {scene_name[17:32]} cfar {len(result['cfar'])} {found} "
            f"gfw {sum(r[2] for r in result['refs'])} {result['seconds']}s",
            flush=True,
        )
    if failed:
        print(f"{failed} passes failed; rerun to retry", flush=True)


def print_summary(summary: dict) -> None:
    if summary["excluded_no_gfw"]:
        print(f"left out {len(summary['excluded_no_gfw'])} passes with no GFW detection in the box")
    for rule, areas in summary["table"].items():
        print(
            f"\n{rule} ({summary['scenes']} passes, {'/'.join(summary['splits'])}, "
            f"thresholds {summary['thresholds']})"
        )
        for area, by_name in areas.items():
            for name, m in by_name.items():
                low, high = m["recall_unmatched_range"]
                print(
                    f"  {area:15s} {name:8s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
                    f"| recall AIS-matched {m['recall_ais_matched']:.3f} unmatched {m['recall_unmatched']:.3f} "
                    f"({low:.3f}-{high:.3f}, n {m['ref_unmatched']})"
                )
    print(
        "\nper model: best threshold on these passes | at threshold T: model - CFAR (95% interval)"
    )
    for label, rules in summary["sweep"].items():
        for rule, best in rules.items():
            line = f"  {label:8s} {rule:15s} best {best['best_threshold']:.2f} F1 {best['best_f1']:.3f}"
            boot = summary["bootstrap"][label][rule]
            if boot:
                line += f" | at {boot['threshold']:.2f}:"
            for key in ("f1", "recall_unmatched") if boot else ():
                low, high = boot[key]["ci95"]
                line += f" | {key} {boot[key]['diff']:+.3f} ({low:+.3f}, {high:+.3f})"
            print(line)
    if "test" in summary["splits"]:
        print("note: thresholds swept on test passes are for information only; choose them on val")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        help=f"LABEL=PATH[@TILE/OVERLAP], repeatable (default model={DEFAULT_MODEL.as_posix()})",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--pairs", type=Path, help='file of "area scene" lines to compare instead of a split'
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=parse_threshold,
        default=[],
        help=f"LABEL=VALUE, repeatable (default {MODEL_THRESHOLD}); choose on val only",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES, help="0 = none")
    parser.add_argument("--index", type=Path, default=Path("data/datasets/index.jsonl"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--per-area", type=int, default=0, help="evenly spaced passes per area (0 = all)"
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    thresholds = dict(args.threshold)

    if not args.summary:
        specs = args.model or [ModelSpec("model", DEFAULT_MODEL)]
        if len({spec.label for spec in specs}) < len(specs):
            parser.error("each --model needs its own label")
        run_passes(args, specs, thresholds)
    if not args.out.is_dir() or not any(p.name != "summary.json" for p in args.out.glob("*.json")):
        print(f"no results in {args.out}; nothing to summarise", flush=True)
        return 1
    summary = summarise(args.out, thresholds=thresholds, resamples=args.bootstrap)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
