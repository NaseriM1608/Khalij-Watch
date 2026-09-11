"""Measure the CFAR false-alarm rate on real open-sea Sentinel-1 clutter.

Completes the Phase 3 definition of done: detections on the map with a
measured false-alarm rate. core/sar/false_alarms.py explains what is measured
and, as importantly, what is not.

Run:
    uv run python scripts/measure_false_alarms.py --lon 57.9 --lat 24.8 --size-km 30

Choose open water. The window is checked against the land mask on a sample
grid and refused if it touches land, because coastline would be booked as
false alarms. That mask deliberately leaves a coastal strip unmasked, so keep
windows well offshore. A JSON record is written to data/evaluation/.

Contains modified Copernicus Sentinel data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.ais.landmask import PolygonLandMask  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.db.spatial import BBox  # noqa: E402
from core.sar.catalogue import SarProduct  # noqa: E402
from core.sar.cfar import CfarConfig  # noqa: E402
from core.sar.false_alarms import (  # noqa: E402
    DEFAULT_EXCLUSION_PX,
    DEFAULT_SWEEP,
    PfaMeasurement,
    measure_clutter_false_alarms,
)
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.reader import S3AccessError, open_measurement, read_with_retry  # noqa: E402
from core.sar.windows import PixelWindow  # noqa: E402
from scripts.detect_window import (  # noqa: E402
    MAX_SIZE_KM,
    box_around,
    latest_product_covering,
    platform_mask,
)

EVALUATION_DIR = ROOT / "data" / "evaluation"
LAND_SAMPLES_PER_SIDE = 25
# A rotated scene's pixel window reaches beyond the box at its corners.
# Sampling the box grown by this fraction of its side on each edge covers it.
LAND_CHECK_MARGIN = 0.25
PIXEL_AREA_KM2 = 1e-4  # 10 m x 10 m


class WindowRefusedError(Exception):
    """The requested window cannot be measured."""


def touches_land(box: BBox) -> bool:
    """Whether a sample grid over the box, grown to cover its pixel window, meets land.

    Decided from coordinates alone, before any imagery is opened. A process
    that opened a scene and closed it without reading was observed to hang on
    exit, so a refused window must never reach object storage.
    """
    lon_margin = (box.max_lon - box.min_lon) * LAND_CHECK_MARGIN
    lat_margin = (box.max_lat - box.min_lat) * LAND_CHECK_MARGIN
    lons = np.linspace(box.min_lon - lon_margin, box.max_lon + lon_margin, LAND_SAMPLES_PER_SIDE)
    lats = np.linspace(box.min_lat - lat_margin, box.max_lat + lat_margin, LAND_SAMPLES_PER_SIDE)
    land = PolygonLandMask()
    return any(land.is_land(float(lon), float(lat)) for lon in lons for lat in lats)


def read_open_water(
    product: SarProduct, polarisation: str, box: BBox
) -> tuple[SceneGeoref, PixelWindow, np.ndarray]:
    """Read the window over ``box``. Land is refused before any imagery is opened."""
    if touches_land(box):
        raise WindowRefusedError(
            "the window touches land; choose open water so coastline is not booked as false alarms"
        )
    try:
        with open_measurement(product, polarisation) as dataset:
            scene = SceneGeoref.from_dataset(dataset)
            window = scene.window_for_bbox(box)
            if window.is_empty():
                raise WindowRefusedError("the requested box does not intersect the scene")
            return scene, window, read_with_retry(dataset, window)
    except S3AccessError as exc:
        raise WindowRefusedError(f"could not read the scene: {exc}") from exc


def print_report(rows: tuple[PfaMeasurement, ...], searched_km2: float) -> None:
    print(
        f"{'Pfa':>7} {'alpha':>6} {'clutter hits':>12} {'measured Pfa':>12} "
        f"{'x request':>9} {'candidates':>10} {'per 100 km2':>11}"
    )
    for row in rows:
        print(
            f"{row.requested_pfa:7.0e} {row.alpha:6.2f} {row.clutter_hits:12d} "
            f"{row.measured_pfa:12.2e} {row.excess:9.2f} {row.candidates:10d} "
            f"{per_100km2(row.candidates, searched_km2):11.2f}"
        )
    clutter_km2 = (rows[0].clutter_pixels if rows else 0) * PIXEL_AREA_KM2
    print(f"clutter area {clutter_km2:.1f} km2 of {searched_km2:.1f} km2 searched")


def per_100km2(count: int, area_km2: float) -> float:
    return count / area_km2 * 100.0 if area_km2 > 0 else float("nan")


def finite_or_none(value: float) -> float | None:
    """JSON has no NaN."""
    return None if math.isnan(value) else value


def build_record(
    args: argparse.Namespace,
    product: SarProduct,
    window: PixelWindow,
    reference: CfarConfig,
    rows: tuple[PfaMeasurement, ...],
    searched_km2: float,
    standing: int,
) -> dict:
    return {
        "product": product.name,
        "acquired": product.acq_start.isoformat(),
        "satellite": product.satellite,
        "polarisation": args.pol.upper(),
        "centre_lon_lat": [args.lon, args.lat],
        "size_km": args.size_km,
        "window": {
            "row_off": window.row_off,
            "col_off": window.col_off,
            "width": window.width,
            "height": window.height,
        },
        "searched_km2": round(searched_km2, 2),
        "standing_platforms_masked": standing,
        "exclusion_px": args.exclusion_px,
        "reference": {
            "guard_half": reference.guard_half,
            "train_half": reference.train_half,
            "looks": reference.looks,
            "pfa": reference.pfa,
        },
        "rows": [
            {
                "requested_pfa": row.requested_pfa,
                "alpha": round(row.alpha, 4),
                "clutter_pixels": row.clutter_pixels,
                "clutter_hits": row.clutter_hits,
                "measured_pfa": finite_or_none(row.measured_pfa),
                "excess": finite_or_none(row.excess),
                "candidates": row.candidates,
                "candidates_per_100km2": finite_or_none(per_100km2(row.candidates, searched_km2)),
            }
            for row in rows
        ],
        "measured_at": dt.datetime.now(dt.UTC).isoformat(),
        "attribution": "Contains modified Copernicus Sentinel data",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--size-km", type=float, default=30.0)
    parser.add_argument("--days", type=int, default=10, help="catalogue lookback")
    parser.add_argument("--pol", choices=("vv", "vh"), default="vh")
    parser.add_argument("--exclusion-px", type=int, default=DEFAULT_EXCLUSION_PX)
    args = parser.parse_args()
    if not 0 < args.size_km <= MAX_SIZE_KM:
        parser.error(f"--size-km must be in (0, {MAX_SIZE_KM}]")
    if args.exclusion_px < 0:
        parser.error("--exclusion-px must be non-negative")

    load_settings()
    box = box_around(args.lon, args.lat, args.size_km)
    product = latest_product_covering(args.lon, args.lat, box, args.days)
    if product is None:
        print(f"no readable Sentinel-1 scene covers {args.lat}, {args.lon} in {args.days} days")
        return 1
    print(f"scene    : {product.name}")
    print(f"acquired : {product.acq_start:%Y-%m-%d %H:%M} UTC ({product.orbit_direction})")

    try:
        scene, window, pixels = read_open_water(product, args.pol, box)
    except WindowRefusedError as exc:
        print(f"refused  : {exc}")
        return 1

    infra, standing = platform_mask(box, product.acq_start.date(), scene, window)
    nodata = pixels == 0
    masked = nodata if infra is None else (nodata | infra)
    searched_km2 = float((~masked).sum()) * PIXEL_AREA_KM2
    print(
        f"window   : {window.width} x {window.height} px ({args.pol.upper()}), "
        f"{searched_km2:.1f} km2 searched, {standing} platforms masked"
    )

    reference = CfarConfig(satellite=product.satellite)
    started = time.perf_counter()
    rows = measure_clutter_false_alarms(
        pixels, mask=masked, pfas=DEFAULT_SWEEP, reference=reference, exclusion_px=args.exclusion_px
    )
    print(f"measured : {len(rows)} settings in {time.perf_counter() - started:.1f}s")
    print()
    print_report(rows, searched_km2)

    record = build_record(args, product, window, reference, rows, searched_km2, standing)
    stem = f"{product.name.removesuffix('.SAFE')}_{args.pol}_{args.lat:.3f}_{args.lon:.3f}"
    path = EVALUATION_DIR / f"false_alarms_{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print()
    print(f"record   : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
