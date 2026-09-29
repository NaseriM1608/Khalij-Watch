"""Mine hard negative chips for an area already built by build_dataset.py.

For every scene in the area's manifest: read VV and VH over the box again,
re-run CFAR to find every target in the window, then cut negative chips at
offshore platforms, on land edges, on bright wind-roughened sea with no point
target, and on extra open sea. Existing manifests are never touched: each
scene's records go to ``<out>/hardneg/records/<stem>.jsonl`` (written
atomically, so a scene counts as done only once its file exists) and its chips
to ``<out>/hardneg/shards/<stem>.npz``. A rerun resumes where the last stopped.

Run:
    uv run python scripts/mine_negatives.py --area-dir data/datasets/fujairah \
        --lon 56.53 --lat 25.22

Exit status 3 means some scenes failed (usually the network); rerun to retry.
Contains modified Copernicus Sentinel data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.config import load_settings  # noqa: E402
from core.db.spatial import BBox  # noqa: E402
from core.labeling.chips import ChipRequest, extract_chips  # noqa: E402
from core.labeling.contact_sheet import contact_sheet  # noqa: E402
from core.labeling.hard_negatives import (  # noqa: E402
    chips_inside,
    clutter_centres,
    coast_edge_centres,
)
from core.labeling.manifest import ChipRecord, records_to_jsonl, split_for  # noqa: E402
from core.labeling.negatives import sample_sea_negatives  # noqa: E402
from core.sar.catalogue import SarProduct  # noqa: E402
from core.sar.cfar import CfarConfig  # noqa: E402
from core.sar.coastline import DEFAULT_BUFFER_M  # noqa: E402
from core.sar.detect import detect_candidates  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.infrastructure import active_platforms, load_platforms  # noqa: E402
from core.sar.reader import S3AccessError  # noqa: E402
from core.sar.windows import PixelWindow  # noqa: E402
from scripts.build_dataset import passes_over, read_pair  # noqa: E402
from scripts.detect_window import (  # noqa: E402
    PLATFORM_PATH,
    box_around,
    coastline_mask,
    platform_mask,
    write_png,
)

# Chip-id prefixes; build_dataset.py uses "d" for detections and "n" for sea negatives.
SOURCE_PREFIX = {"platform": "hp", "coast": "hc", "clutter": "hk", "sea": "hs"}
SHEET_CHIPS = 64
MINER = "hardneg-v1"


def write_scene_records(directory: Path, stem: str, records: Sequence[ChipRecord]) -> Path:
    """Write one scene's records atomically: the file exists only when complete."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.jsonl"
    temporary = directory / f"{stem}.jsonl.tmp"
    temporary.write_text(records_to_jsonl(records), encoding="utf-8")
    temporary.replace(path)
    return path


def done_scenes(directory: Path) -> set[str]:
    """Stems of scenes whose records were written completely."""
    if not directory.is_dir():
        return set()
    return {p.name.removesuffix(".jsonl") for p in directory.glob("*.jsonl")}


def hard_requests(
    stem: str, groups: dict[str, list[tuple[float, float]]], *, window_offset: tuple[int, int]
) -> list[tuple[ChipRequest, str]]:
    """Chip requests in scene coordinates, named by source, from window centres."""
    row_off, col_off = window_offset
    requests = []
    for source, centres in groups.items():
        prefix = SOURCE_PREFIX[source]
        for i, (row, col) in enumerate(centres):
            request = ChipRequest(f"{stem}/{prefix}{i:04d}", row + row_off, col + col_off)
            requests.append((request, source))
    return requests


def platform_window_points(
    rows: Sequence[float], cols: Sequence[float], *, window_offset: tuple[int, int]
) -> list[tuple[float, float]]:
    """Corner-based scene raster coordinates to pixel-centre window coordinates."""
    row_off, col_off = window_offset
    return [
        (float(r) - 0.5 - row_off, float(c) - 0.5 - col_off)
        for r, c in zip(rows, cols, strict=True)
    ]


def platform_centres(
    box: BBox, when: dt.date, scene: SceneGeoref, window: PixelWindow
) -> list[tuple[float, float]]:
    """Window centres of offshore structures standing on the acquisition date."""
    if not PLATFORM_PATH.is_file():
        return []
    standing = active_platforms(load_platforms(PLATFORM_PATH, region=None, bbox=box), when)
    if not standing:
        return []
    rows, cols = scene.raster_coords([p.lon for p in standing], [p.lat for p in standing])
    points = platform_window_points(rows, cols, window_offset=(window.row_off, window.col_off))
    return chips_inside(points, (window.height, window.width))


def scene_rng(stem: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{MINER}:{stem}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def centre_groups(
    product: SarProduct,
    box: BBox,
    scene: SceneGeoref,
    window: PixelWindow,
    bands: np.ndarray,
    known: dict[str, list[tuple[float, float]]],
    args: argparse.Namespace,
) -> dict[str, list[tuple[float, float]]]:
    """Window centres of every hard negative kind for one scene."""
    stem = product.name.removesuffix(".SAFE")
    nodata = (bands == 0).any(axis=0)
    masked = nodata.copy()
    for layer in (
        platform_mask(box, product.acq_start.date(), scene, window)[0],
        coastline_mask(box, scene, window, DEFAULT_BUFFER_M),
    ):
        if layer is not None:
            masked |= layer
    land = coastline_mask(box, scene, window, 0.0)

    offset = (window.row_off, window.col_off)
    config = CfarConfig(pfa=args.pfa, satellite=product.satellite)
    found = [(c.row, c.col) for c in detect_candidates(bands[1], config=config, mask=masked)]
    targets = found + [(r - offset[0], c - offset[1]) for r, c in known["detections"]]
    sea_taken = [(r - offset[0], c - offset[1]) for r, c in known["negatives"]]

    rng = scene_rng(stem)
    groups = {"platform": platform_centres(box, product.acq_start.date(), scene, window)}
    groups["coast"] = (
        coast_edge_centres(land, nodata, targets, count=args.coast, rng=rng)
        if land is not None
        else []
    )
    groups["clutter"] = clutter_centres(bands[1], masked, targets, count=args.clutter, rng=rng)
    groups["sea"] = sample_sea_negatives(masked, targets + sea_taken, count=args.sea, rng=rng)
    return groups


def mine_scene(
    product: SarProduct,
    box: BBox,
    known: dict[str, list[tuple[float, float]]],
    args: argparse.Namespace,
    out: Path,
) -> Counter:
    """Cut, store and record one scene's hard negatives; returns counts by source."""
    stem = product.name.removesuffix(".SAFE")
    scene, window, bands = read_pair(product, box)
    groups = centre_groups(product, box, scene, window, bands, known, args)
    requests = hard_requests(stem, groups, window_offset=(window.row_off, window.col_off))
    source_of = {r.chip_id: s for r, s in requests}
    by_id = {r.chip_id: r for r, _ in requests}
    chips, kept, _ = extract_chips(
        bands, [r for r, _ in requests], window_offset=(window.row_off, window.col_off)
    )

    rows = [by_id[i].row for i in kept]
    cols = [by_id[i].col for i in kept]
    lons, lats = scene.lonlat(rows, cols) if kept else ([], [])
    shard = f"hardneg/shards/{stem}.npz"
    records = [
        ChipRecord(
            chip_id=chip_id,
            scene=product.name,
            split=split_for(product.name),
            shard=shard,
            index=index,
            lon=float(lons[index]),
            lat=float(lats[index]),
            row=rows[index],
            col=cols[index],
            kind="negative",
            source=source_of[chip_id],
            extra={"miner": MINER},
        )
        for index, chip_id in enumerate(kept)
    ]
    write_shard(out / shard, chips, kept)
    write_sheets(out / "hardneg" / "contact", stem, chips, records)
    write_scene_records(out / "hardneg" / "records", stem, records)
    return Counter(r.source for r in records)


def write_shard(path: Path, chips: np.ndarray, kept: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, chips=chips, chip_ids=np.array(kept))
    temporary.replace(path)


def write_sheets(directory: Path, stem: str, chips: np.ndarray, records: list[ChipRecord]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for source in SOURCE_PREFIX:
        indices = [r.index for r in records if r.source == source][:SHEET_CHIPS]
        if indices:
            write_png(directory / f"{stem}_{source}.png", contact_sheet(chips[indices], columns=8))


def known_positions(manifest: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Scene (row, col) of each scene's detections and existing sea negatives."""
    known: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {"detections": [], "negatives": []}
    )
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = "negatives" if r["kind"] == "negative" else "detections"
        known[r["scene"]][key].append((r["row"], r["col"]))
    return known


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area-dir", type=Path, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--size-km", type=float, default=20.0)
    parser.add_argument("--since", type=dt.date.fromisoformat, default=dt.date(2017, 1, 1))
    parser.add_argument("--until", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--pfa", type=float, default=1e-6)
    parser.add_argument("--coast", type=int, default=24, help="land-edge chips per scene")
    parser.add_argument("--clutter", type=int, default=32, help="bright-sea chips per scene")
    parser.add_argument("--sea", type=int, default=16, help="extra open-sea chips per scene")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many scenes")
    args = parser.parse_args()

    load_settings()
    known = known_positions(args.area_dir / "manifest.jsonl")
    box = box_around(args.lon, args.lat, args.size_km)
    since = dt.datetime.combine(args.since, dt.time(), dt.UTC)
    until = dt.datetime.combine(args.until, dt.time(), dt.UTC)
    done = done_scenes(args.area_dir / "hardneg" / "records")
    todo = [
        p
        for p in passes_over(box, since, until)
        if p.name in known and p.name.removesuffix(".SAFE") not in done
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(known)} scenes in the manifest, {len(done)} already mined, {len(todo)} to mine")

    totals: Counter = Counter()
    failed = []
    for product in todo:
        started = time.perf_counter()
        try:
            counts = mine_scene(product, box, known[product.name], args, args.area_dir)
        except (S3AccessError, ValueError) as exc:
            failed.append(product.name)
            print(f"  {product.name.removesuffix('.SAFE')}: FAILED {str(exc)[:100]}", flush=True)
            continue
        totals.update(counts)
        print(
            f"  {product.name.removesuffix('.SAFE')}: {dict(counts)} "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )
    print(f"totals: {dict(totals)}; failed scenes: {len(failed)}")
    print("Contains modified Copernicus Sentinel data.")
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
