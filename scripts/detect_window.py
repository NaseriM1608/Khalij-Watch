"""Run the CFAR detector on a real Sentinel-1 window and show what it found.

The first end-to-end run of the Phase 3 detector on real imagery: catalogue
-> object storage -> ground-control-point georeferencing -> platform mask ->
CFAR -> vessel candidates. Results are printed as a table and written as a
GeoJSON file plus an annotated PNG for checking by eye.

Run:
    uv run python scripts/detect_window.py --lon 56.75 --lat 25.25 --size-km 20

Choose an offshore window for now. The coastline mask is still the
conservative interior polygon set from Phase 1.5, which deliberately leaves
the coastal strip unmasked, so shoreline inside the window will be reported
as candidates until a proper coastline is added.

If CDSE rejects the credentials this process may not exit on its own: GDAL
hangs in native teardown after a failed request on this host. Run
scripts/check_access.py first, which isolates that failure.

Contains modified Copernicus Sentinel data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.ais.landmask import PolygonLandMask, point_in_polygon  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.db.spatial import BBox  # noqa: E402
from core.sar.catalogue import SarProduct, search  # noqa: E402
from core.sar.cfar import CfarConfig  # noqa: E402
from core.sar.detect import Candidate, detect_candidates, georeference  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.infrastructure import (  # noqa: E402
    ATTRIBUTION as PLATFORM_ATTRIBUTION,
)
from core.sar.infrastructure import (  # noqa: E402
    DEFAULT_PATH as PLATFORM_PATH,
)
from core.sar.infrastructure import (  # noqa: E402
    InfrastructureMask,
    load_platforms,
)
from core.sar.reader import S3AccessError, open_measurement, read_with_retry  # noqa: E402
from core.sar.windows import PixelWindow  # noqa: E402

KM_PER_DEG_LAT = 110.54
KM_PER_DEG_LON_EQUATOR = 111.32
MAX_SIZE_KM = 60.0

QUICKLOOK_DIR = ROOT / "data" / "quicklooks"
DETECTION_DIR = ROOT / "data" / "detections"
PREVIEW_MAX_SIDE = 1600
MARKER_HALF = 7


def box_around(lon: float, lat: float, size_km: float) -> BBox:
    half = size_km / 2.0
    dlat = half / KM_PER_DEG_LAT
    dlon = half / (KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)))
    return BBox(lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def latest_product_covering(lon: float, lat: float, box: BBox, days: int) -> SarProduct | None:
    """The most recent readable scene whose footprint contains the point."""
    since = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    covering = [
        p
        for p in search(box, since=since)
        if p.is_cog and p.online and p.s3_path and point_in_polygon(lon, lat, p.footprint)
    ]
    return max(covering, key=lambda p: p.acq_start) if covering else None


def platform_mask(
    box: BBox, when: dt.date, scene: SceneGeoref, window: PixelWindow
) -> tuple[np.ndarray | None, int]:
    """Rasterise structures standing on the acquisition date, if the dataset is present."""
    if not PLATFORM_PATH.is_file():
        return None, 0
    margin = 0.05  # a little beyond the window, so edge structures still mask
    expanded = BBox(
        box.min_lon - margin, box.min_lat - margin, box.max_lon + margin, box.max_lat + margin
    )
    # Every region, not just the Persian Gulf: the window may sit outside it.
    standing = InfrastructureMask(load_platforms(region=None, bbox=expanded)).for_date(when)
    raster = standing.rasterise(
        scene, window.width, window.height, window_offset=(window.row_off, window.col_off)
    )
    return raster, len(standing)


def display_stretch(pixels: np.ndarray) -> np.ndarray:
    """Percentile stretch for viewing only.

    Deliberately not the frozen sigmoid used as model input: this exists to
    make faint vessels visible to a person, and must never feed a model.
    """
    valid = pixels[pixels > 0]
    if valid.size == 0:
        return np.zeros(pixels.shape, dtype=np.uint8)
    low, high = np.percentile(valid, [2.0, 99.8])
    scaled = np.clip((pixels.astype(np.float32) - low) / max(high - low, 1.0), 0.0, 1.0)
    return (np.sqrt(scaled) * 255.0).astype(np.uint8)


def draw_box(rgb: np.ndarray, row: int, col: int, half: int) -> None:
    """A hollow red square, clipped to the image."""
    _, height, width = rgb.shape
    top, bottom = max(0, row - half), min(height - 1, row + half)
    left, right = max(0, col - half), min(width - 1, col + half)
    if top > bottom or left > right:
        return
    for r0, r1, c0, c1 in (
        (top, top + 1, left, right + 1),
        (bottom, bottom + 1, left, right + 1),
        (top, bottom + 1, left, left + 1),
        (top, bottom + 1, right, right + 1),
    ):
        rgb[0, r0:r1, c0:c1] = 255
        rgb[1, r0:r1, c0:c1] = 40
        rgb[2, r0:r1, c0:c1] = 40


def render_preview(
    pixels: np.ndarray, detections: list[Candidate], window: PixelWindow, masked: np.ndarray
) -> np.ndarray:
    stride = max(1, math.ceil(max(pixels.shape) / PREVIEW_MAX_SIDE))
    grey = display_stretch(pixels[::stride, ::stride])
    rgb = np.stack([grey, grey, grey])
    shown = masked[::stride, ::stride]
    rgb[0][shown] //= 3
    rgb[1][shown] //= 3  # masked pixels tinted blue
    for candidate in detections:
        row = round((candidate.row - window.row_off) / stride)
        col = round((candidate.col - window.col_off) / stride)
        draw_box(rgb, row, col, MARKER_HALF)
    return rgb


def write_png(path: Path, rgb: np.ndarray) -> None:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path, "w", driver="PNG", width=rgb.shape[2], height=rgb.shape[1], count=3, dtype="uint8"
        ) as out:
            out.write(rgb)


def write_geojson(path: Path, product: SarProduct, detections: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
            "properties": {
                "length_m": round(c.length_m, 1),
                "width_m": round(c.width_m, 1),
                "heading_deg": round(c.heading_deg, 1),
                "pixel_count": c.pixel_count,
                "peak_dn": round(c.peak_amplitude, 1),
                "detector": "cfar",
            },
        }
        for c in detections
    ]
    collection = {
        "type": "FeatureCollection",
        "properties": {
            "product": product.name,
            "acquired": product.acq_start.isoformat(),
            "attribution": "Contains modified Copernicus Sentinel data",
        },
        "features": features,
    }
    path.write_text(json.dumps(collection, indent=1), encoding="utf-8")


def print_table(detections: list[Candidate], limit: int = 25) -> None:
    print(f"{'lat':>9} {'lon':>9} {'length m':>9} {'width m':>8} {'heading':>8} {'peak DN':>8}")
    for c in sorted(detections, key=lambda d: -d.peak_amplitude)[:limit]:
        print(
            f"{c.lat:9.4f} {c.lon:9.4f} {c.length_m:9.0f} {c.width_m:8.0f} "
            f"{c.heading_deg:7.0f}d {c.peak_amplitude:8.0f}"
        )
    if len(detections) > limit:
        print(f"... and {len(detections) - limit} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--size-km", type=float, default=20.0)
    parser.add_argument("--days", type=int, default=10, help="catalogue lookback")
    parser.add_argument("--pol", choices=("vv", "vh"), default="vh")
    parser.add_argument("--pfa", type=float, default=1e-6)
    args = parser.parse_args()
    if not 0 < args.size_km <= MAX_SIZE_KM:
        parser.error(f"--size-km must be in (0, {MAX_SIZE_KM}]")

    load_settings()
    box = box_around(args.lon, args.lat, args.size_km)
    product = latest_product_covering(args.lon, args.lat, box, args.days)
    if product is None:
        print(f"no readable Sentinel-1 scene covers {args.lat}, {args.lon} in {args.days} days")
        return 1
    print(f"scene    : {product.name}")
    print(f"acquired : {product.acq_start:%Y-%m-%d %H:%M} UTC ({product.orbit_direction})")

    try:
        with open_measurement(product, args.pol) as dataset:
            scene = SceneGeoref.from_dataset(dataset)
            window = scene.window_for_bbox(box)
            if window.is_empty():
                print("the requested box does not intersect the scene")
                return 1
            started = time.perf_counter()
            pixels = read_with_retry(dataset, window)
            read_s = time.perf_counter() - started
    except S3AccessError as exc:
        print(f"could not read the scene: {exc}")
        return 1

    print(
        f"window   : {window.width} x {window.height} px ({args.pol.upper()}), "
        f"read in {read_s:.1f}s"
    )

    infra, standing = platform_mask(box, product.acq_start.date(), scene, window)
    nodata = pixels == 0
    masked = nodata if infra is None else (nodata | infra)
    print(
        f"masking  : {standing} standing platforms, {infra.sum() if infra is not None else 0} px; "
        f"nodata {int(nodata.sum())} px"
    )

    config = CfarConfig(pfa=args.pfa, satellite=product.satellite)
    started = time.perf_counter()
    candidates = detect_candidates(pixels, config=config, mask=masked)
    located = georeference(candidates, scene, window_offset=(window.row_off, window.col_off))
    detect_s = time.perf_counter() - started

    land = PolygonLandMask()
    offshore = [c for c in located if not land.is_land(c.lon, c.lat)]
    area_km2 = (window.width * 10.0 / 1000.0) * (window.height * 10.0 / 1000.0)
    print(
        f"detector : CFAR Pfa {args.pfa:g} (alpha {config.alpha:.2f}) in {detect_s:.1f}s -> "
        f"{len(offshore)} candidates offshore ({len(located) - len(offshore)} inland dropped), "
        f"{len(offshore) / max(area_km2, 1e-6) * 100:.1f} per 100 km2"
    )
    print()
    print_table(offshore)

    stem = f"{product.name.removesuffix('.SAFE')}_{args.pol}_{args.lat:.3f}_{args.lon:.3f}"
    png = QUICKLOOK_DIR / f"{stem}.png"
    geojson = DETECTION_DIR / f"{stem}.geojson"
    write_png(png, render_preview(pixels, offshore, window, masked))
    write_geojson(geojson, product, offshore)
    print()
    print(f"preview  : {png}")
    print(f"geojson  : {geojson}")
    if infra is not None:
        print(PLATFORM_ATTRIBUTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
