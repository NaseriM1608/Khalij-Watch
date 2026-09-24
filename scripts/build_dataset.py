"""Build an AIS-labelled chip dataset for one area from Global Fishing Watch matches.

For every readable Sentinel-1 pass over the box in the period:
    read VV and VH -> mask nodata, platforms and coastline -> CFAR on VH ->
    georeference -> associate each detection with GFW's AIS-matched cells ->
    cut 128x128 chips for every detection plus open-sea negatives ->
    write one shard per scene, update the manifest, render contact sheets.

Labels come from Global Fishing Watch (non-commercial use) and need
GFW_API_TOKEN in .env. Prefer periods before 2026: at Fujairah GFW matched 99%
of its detections to AIS in 2024-2025 but only 16-53% in 2026.

Run:
    uv run --with gfw-api-python-client python scripts/build_dataset.py \
        --lon 56.53 --lat 25.22 --since 2025-09-01 --until 2025-10-01 \
        --out data/datasets/fujairah

Contains modified Copernicus Sentinel data.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.ais.landmask import point_in_polygon  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.db.spatial import BBox  # noqa: E402
from core.labeling.chips import ChipRequest, extract_chips  # noqa: E402
from core.labeling.contact_sheet import contact_sheet  # noqa: E402
from core.labeling.gfw import GfwCell, associate, cells_for_pass, label_fields  # noqa: E402
from core.labeling.gfw_api import fetch_sar_cells  # noqa: E402
from core.labeling.manifest import (  # noqa: E402
    ChipRecord,
    records_from_jsonl,
    records_to_jsonl,
    split_for,
)
from core.labeling.negatives import sample_sea_negatives  # noqa: E402
from core.labeling.scenes import one_per_pass  # noqa: E402
from core.sar.catalogue import SarProduct, search  # noqa: E402
from core.sar.cfar import CfarConfig  # noqa: E402
from core.sar.coastline import DEFAULT_BUFFER_M  # noqa: E402
from core.sar.detect import Candidate, detect_candidates, georeference  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.reader import S3AccessError, open_measurement, read_with_retry  # noqa: E402
from core.sar.windows import PixelWindow  # noqa: E402
from scripts.detect_window import (  # noqa: E402
    box_around,
    coastline_mask,
    platform_mask,
    write_png,
)

SHEET_CHIPS = 64


@dataclass
class SceneResult:
    """What one scene contributed."""

    scene: str
    acquired: str
    records: list[ChipRecord] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    # Why a readable scene contributed nothing, such as GFW never processing it.
    skipped: str | None = None


def passes_over(box: BBox, since: dt.datetime, until: dt.datetime) -> list[SarProduct]:
    """One readable product per pass whose footprint holds the box centre."""
    lon = (box.min_lon + box.max_lon) / 2.0
    lat = (box.min_lat + box.max_lat) / 2.0
    return one_per_pass(
        p
        for p in search(box, since=since, until=until)
        if p.is_cog and p.online and p.s3_path and point_in_polygon(lon, lat, p.footprint)
    )


def read_pair(product: SarProduct, box: BBox) -> tuple[SceneGeoref, PixelWindow, np.ndarray]:
    """VV and VH over the box as one (2, rows, cols) stack, on the same pixel grid."""
    with open_measurement(product, "vh") as dataset:
        scene = SceneGeoref.from_dataset(dataset)
        window = scene.window_for_bbox(box)
        vh = read_with_retry(dataset, window)
    with open_measurement(product, "vv") as dataset:
        vv = read_with_retry(dataset, window)
    if vv.shape != vh.shape:
        raise ValueError(f"VV {vv.shape} and VH {vh.shape} windows differ")
    return scene, window, np.stack([vv, vh])


def exclusion_mask(
    box: BBox,
    product: SarProduct,
    scene: SceneGeoref,
    window: PixelWindow,
    bands: np.ndarray,
    buffer_m: float,
) -> np.ndarray:
    masked = (bands == 0).any(axis=0)
    for layer in (
        platform_mask(box, product.acq_start.date(), scene, window)[0],
        coastline_mask(box, scene, window, buffer_m),
    ):
        if layer is not None:
            masked |= layer
    return masked


def scene_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def negative_positions(
    masked: np.ndarray,
    scene: SceneGeoref,
    window: PixelWindow,
    detections: list[Candidate],
    gfw_cells: list[GfwCell],
    count: int,
    seed: int,
) -> list[tuple[float, float]]:
    """Scene (row, col) of open-sea negatives, clear of our detections and of GFW's."""
    targets = [(c.row - window.row_off, c.col - window.col_off) for c in detections]
    if gfw_cells:
        rows, cols = scene.raster_coords([c.lon for c in gfw_cells], [c.lat for c in gfw_cells])
        targets += [
            (r - window.row_off, c - window.col_off) for r, c in zip(rows, cols, strict=True)
        ]
    found = sample_sea_negatives(masked, targets, count=count, rng=np.random.default_rng(seed))
    return [(r + window.row_off, c + window.col_off) for r, c in found]


def detection_extra(candidate: Candidate) -> dict:
    return {
        "cfar_length_m": round(candidate.length_m, 1),
        "cfar_width_m": round(candidate.width_m, 1),
        "cfar_heading_deg": round(candidate.heading_deg, 1),
        "peak_dn": round(candidate.peak_amplitude, 1),
        "pixel_count": candidate.pixel_count,
    }


def process_scene(
    product: SarProduct, box: BBox, cells: list[GfwCell], args: argparse.Namespace, out: Path
) -> SceneResult:
    stem = product.name.removesuffix(".SAFE")
    result = SceneResult(scene=product.name, acquired=product.acq_start.isoformat())
    pass_cells = cells_for_pass(cells, product.acq_start)
    if not pass_cells:
        # GFW never processed this pass (S1C throughout 2025, for one), so
        # "GFW saw nothing" would be false. Skip it before reading any pixels.
        result.skipped = "no GFW detections for this pass"
        remove_outputs(out, stem)
        return result
    try:
        scene, window, bands = read_pair(product, box)
    except (S3AccessError, ValueError) as exc:
        result.error = str(exc)
        return result

    masked = exclusion_mask(box, product, scene, window, bands, args.coast_buffer_m)
    config = CfarConfig(pfa=args.pfa, satellite=product.satellite)
    located = georeference(
        detect_candidates(bands[1], config=config, mask=masked),
        scene,
        window_offset=(window.row_off, window.col_off),
    )
    detections = [c for c in located if box.contains(c.lon, c.lat)]
    assignments = associate(pass_cells, [(c.lon, c.lat) for c in detections])

    requests = [ChipRequest(f"{stem}/d{i:04d}", c.row, c.col) for i, c in enumerate(detections)]
    sea = negative_positions(
        masked, scene, window, located, pass_cells, args.negatives, scene_seed(stem)
    )
    requests += [ChipRequest(f"{stem}/n{i:04d}", r, c) for i, (r, c) in enumerate(sea)]
    chips, kept, skipped = extract_chips(
        bands, requests, window_offset=(window.row_off, window.col_off)
    )

    by_id = {
        f"{stem}/d{i:04d}": (c, a)
        for i, (c, a) in enumerate(zip(detections, assignments, strict=True))
    }
    sea_by_id = {f"{stem}/n{i:04d}": rc for i, rc in enumerate(sea)}
    split = split_for(product.name)
    shard = f"shards/{stem}.npz"
    for index, chip_id in enumerate(kept):
        if chip_id in by_id:
            candidate, assignment = by_id[chip_id]
            fields = label_fields(assignment)
            fields["extra"] = {**fields["extra"], **detection_extra(candidate)}
            lon, lat, row, col = candidate.lon, candidate.lat, candidate.row, candidate.col
        else:
            row, col = sea_by_id[chip_id]
            lons, lats = scene.lonlat([row], [col])
            lon, lat = float(lons[0]), float(lats[0])
            fields = {
                "kind": "negative",
                "source": "sea",
                "mmsi": None,
                "vessel_type": None,
                "extra": {},
            }
        result.records.append(
            ChipRecord(
                chip_id=chip_id,
                scene=product.name,
                split=split,
                shard=shard,
                index=index,
                lon=lon,
                lat=lat,
                row=row,
                col=col,
                **fields,
            )
        )

    (out / "shards").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / shard, chips=chips, chip_ids=np.array(kept))
    write_sheets(out, stem, chips, result.records)
    statuses = [r.extra.get("gfw_status") for r in result.records if r.kind != "negative"]
    reasons = [r.extra.get("gfw_reason") for r in result.records if r.kind != "negative"]
    result.counts = {
        "detections": len(detections),
        "labelled": statuses.count("labelled"),
        "gfw_unmatched": statuses.count("gfw_unmatched"),
        "gfw_absent": statuses.count("gfw_absent"),
        "ambiguous": statuses.count("ambiguous"),
        "ambiguous_edge": reasons.count("edge"),
        "ambiguous_crowded_ours": reasons.count("crowded_ours"),
        "ambiguous_crowded_gfw": reasons.count("crowded_gfw"),
        "negatives": sum(1 for r in result.records if r.kind == "negative"),
        "skipped": len(skipped),
        "gfw_matched_cells": sum(1 for c in pass_cells if c.matched),
        "gfw_unmatched_detections": sum(c.detections for c in pass_cells if not c.matched),
    }
    return result


def remove_outputs(out: Path, stem: str) -> None:
    """Delete a scene's shard and sheets left by an earlier build."""
    (out / "shards" / f"{stem}.npz").unlink(missing_ok=True)
    for name in ("labelled", "unlabelled", "negatives"):
        (out / "contact" / f"{stem}_{name}.png").unlink(missing_ok=True)


def write_sheets(out: Path, stem: str, chips: np.ndarray, records: list[ChipRecord]) -> None:
    (out / "contact").mkdir(parents=True, exist_ok=True)
    for name, wanted in (
        ("labelled", "vessel"),
        ("unlabelled", "unlabelled"),
        ("negatives", "negative"),
    ):
        indices = [r.index for r in records if r.kind == wanted][:SHEET_CHIPS]
        if indices:
            write_png(
                out / "contact" / f"{stem}_{name}.png", contact_sheet(chips[indices], columns=8)
            )


def update_manifest(path: Path, results: list[SceneResult]) -> int:
    """Replace the records of every processed scene, keep the rest."""
    existing = records_from_jsonl(path.read_text(encoding="utf-8")) if path.is_file() else []
    processed = {r.scene for r in results if r.error is None}
    records = [r for r in existing if r.scene not in processed]
    records += [record for r in results for record in r.records]
    path.write_text(records_to_jsonl(records), encoding="utf-8")
    return len(records)


def print_summary(results: list[SceneResult]) -> None:
    print(
        f"\n{'scene':62s} {'split':5s} {'det':>4} {'lab':>4} {'unm':>4} {'abs':>4} {'amb':>4} {'neg':>4}"
    )
    totals: dict[str, int] = {}
    for result in results:
        if result.error:
            print(f"{result.scene:62s} FAILED: {result.error[:60]}")
            continue
        if result.skipped:
            print(f"{result.scene.removesuffix('.SAFE')[:62]:62s} skipped: {result.skipped}")
            continue
        c = result.counts
        for key, value in c.items():
            totals[key] = totals.get(key, 0) + value
        print(
            f"{result.scene.removesuffix('.SAFE')[:62]:62s} {split_for(result.scene):5s} "
            f"{c['detections']:4d} {c['labelled']:4d} {c['gfw_unmatched']:4d} "
            f"{c['gfw_absent']:4d} {c['ambiguous']:4d} {c['negatives']:4d}"
        )
    print(f"totals: {totals}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--size-km", type=float, default=20.0)
    parser.add_argument("--since", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--until", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pfa", type=float, default=1e-6)
    parser.add_argument("--negatives", type=int, default=16, help="open-sea chips per scene")
    parser.add_argument("--coast-buffer-m", type=float, default=DEFAULT_BUFFER_M)
    args = parser.parse_args()

    load_settings()
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        print("GFW_API_TOKEN is not set; get one at https://globalfishingwatch.org/our-apis/tokens")
        return 1
    import gfwapiclient

    box = box_around(args.lon, args.lat, args.size_km)
    since = dt.datetime.combine(args.since, dt.time(), dt.UTC)
    until = dt.datetime.combine(args.until, dt.time(), dt.UTC)
    products = passes_over(box, since, until)
    print(f"{len(products)} passes over the box between {args.since} and {args.until}")

    cells = asyncio.run(
        fetch_sar_cells(gfwapiclient.Client(access_token=token), box, args.since, args.until)
    )
    print(
        f"GFW cells: {sum(c.matched for c in cells)} matched, {sum(not c.matched for c in cells)} unmatched"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for product in products:
        started = time.perf_counter()
        results.append(process_scene(product, box, cells, args, args.out))
        print(
            f"  {product.name.removesuffix('.SAFE')}: {time.perf_counter() - started:.0f}s",
            flush=True,
        )

    total = update_manifest(args.out / "manifest.jsonl", results)
    (args.out / "summary.json").write_text(
        json.dumps(
            [
                {
                    "scene": r.scene,
                    "acquired": r.acquired,
                    "error": r.error,
                    "skipped": r.skipped,
                    **r.counts,
                }
                for r in results
            ],
            indent=1,
        ),
        encoding="utf-8",
    )
    print_summary(results)
    print(f"manifest: {total} records in {args.out / 'manifest.jsonl'}")
    print(
        "Labels: Global Fishing Watch (non-commercial). Contains modified Copernicus Sentinel data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
