"""Run the CFAR detector on a real Sentinel-1 window and show what it found.

The first end-to-end run of the Phase 3 detector on real imagery: catalogue
-> object storage -> ground-control-point georeferencing -> platform mask ->
CFAR -> vessel candidates. Results are printed as a table, written as a
GeoJSON file plus an annotated PNG for checking by eye, and stored in the
database for the API and map. Storing replaces any earlier run over the same
box, scene and polarisation; --no-store skips it.

Run:
    uv run python scripts/detect_window.py --lon 56.75 --lat 25.25 --size-km 20

Land and a seaward buffer are masked before CFAR using the OpenStreetMap
coastline built by scripts/fetch_coastline.py. Without that file the run
falls back to dropping candidates inside the conservative interior polygons
from Phase 1.5, which leave the coastal strip unmasked, so shoreline would be
reported as candidates.

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
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.ais.landmask import PolygonLandMask, point_in_polygon  # noqa: E402

# Other scripts import box_around from here.
from core.areas import box_around  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.db.repository import (  # noqa: E402
    AnalysedWindowRepository,
    DetectionRepository,
    SceneRepository,
)
from core.db.session import (  # noqa: E402
    create_all,
    create_db_engine,
    make_session_factory,
    session_scope,
)
from core.db.spatial import BBox  # noqa: E402
from core.sar.catalogue import SarProduct, search  # noqa: E402
from core.sar.cfar import CfarConfig  # noqa: E402
from core.sar.coastline import (  # noqa: E402
    ATTRIBUTION as COASTLINE_ATTRIBUTION,
)
from core.sar.coastline import (  # noqa: E402
    DEFAULT_BUFFER_M,
    CoastlineMask,
    load_land,
)
from core.sar.coastline import (  # noqa: E402
    DEFAULT_PATH as COASTLINE_PATH,
)
from core.sar.detect import (  # noqa: E402
    CFAR_CONFIDENCE,
    Candidate,
    detect_candidates,
    detector_version,
    georeference,
)
from core.sar.display import display_stretch  # noqa: E402
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

MAX_SIZE_KM = 60.0

QUICKLOOK_DIR = ROOT / "data" / "quicklooks"
DETECTION_DIR = ROOT / "data" / "detections"
PREVIEW_MAX_SIDE = 1600
MARKER_HALF = 7


def choose_product(
    products: Sequence[SarProduct], lon: float, lat: float, scene: str | None = None
) -> SarProduct | None:
    """The newest readable scene whose footprint contains the point; with ``scene``, that one only."""
    covering = [
        p
        for p in products
        if p.is_cog
        and p.online
        and p.s3_path
        and point_in_polygon(lon, lat, p.footprint)
        and (scene is None or p.name == scene)
    ]
    return max(covering, key=lambda p: p.acq_start) if covering else None


def latest_product_covering(
    lon: float, lat: float, box: BBox, days: int, scene: str | None = None
) -> SarProduct | None:
    """The most recent readable scene whose footprint contains the point (or the named one)."""
    since = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    return choose_product(search(box, since=since), lon, lat, scene)


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


def coastline_mask(
    box: BBox, scene: SceneGeoref, window: PixelWindow, buffer_m: float
) -> np.ndarray | None:
    """Rasterise land and its seaward buffer, if the coastline file is present."""
    if not COASTLINE_PATH.is_file():
        return None
    # Wide enough for the buffer and for a rotated window's corners beyond the box.
    margin = 0.25 * max(box.max_lon - box.min_lon, box.max_lat - box.min_lat) + 0.02
    nearby = BBox(
        box.min_lon - margin, box.min_lat - margin, box.max_lon + margin, box.max_lat + margin
    )
    land = load_land(COASTLINE_PATH, bbox=nearby)
    return CoastlineMask(land, buffer_m=buffer_m).rasterise(
        scene, window.width, window.height, window_offset=(window.row_off, window.col_off)
    )


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


def write_geojson(
    path: Path, product: SarProduct, detections: list[Candidate], version: str
) -> None:
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
            "detector_version": version,
            "attribution": "Contains modified Copernicus Sentinel data",
        },
        "features": features,
    }
    path.write_text(json.dumps(collection, indent=1), encoding="utf-8")


def store_detections(
    database_url: str | None,
    product: SarProduct,
    box: BBox,
    detections: list[Candidate],
    polarisation: str,
    version: str,
    *,
    detector: str = "cfar",
    confidence: float = CFAR_CONFIDENCE,
) -> int:
    """Record the scene, replace this box's detections from it, and record the box as analysed.

    The window is recorded even when nothing was found: it is what lets the
    console tell searched water from water nobody searched (PLAN §10).
    """
    engine = create_db_engine(database_url)
    create_all(engine)
    with session_scope(make_session_factory(engine)) as session:
        scenes = SceneRepository(session)
        scenes.upsert(product, status="discovered" if product.online else "offline")
        session.flush()
        scene_id = scenes.id_for(product.product_id)
        if scene_id is None:
            raise RuntimeError(f"scene {product.product_id} was not recorded")
        written = DetectionRepository(session).replace_in_bbox(
            scene_id,
            box,
            detections,
            polarisation=polarisation.upper(),
            detector=detector,
            detector_version=version,
            confidence=confidence,
        )
        AnalysedWindowRepository(session).record(
            scene_id,
            box,
            detector=detector,
            detector_version=version,
            polarisation=polarisation.upper(),
            detections=written,
        )
        return written


def cut_found_chips(
    found: Sequence[object], scene: SceneGeoref, window: PixelWindow, bands, product, root: Path
) -> None:
    """Each stored detection's satellite-image chip, cut from the pixels the model searched.

    Chips reaching beyond those pixels are read from the scene (opened once);
    a chip failure never fails the detection run. scripts/cut_chips.py.
    """
    from core.inference.contract import BAND_ORDER
    from scripts.cut_chips import HeldBand, cut_detection_chips, open_scene_bands

    held = {
        pol: HeldBand(bands[i], row_off=window.row_off, col_off=window.col_off)
        for i, pol in enumerate(BAND_ORDER)
    }
    done = cut_detection_chips(
        [(d.lon, d.lat) for d in found],
        held,
        scene,
        root=root,
        scene=product.name,
        open_scene=lambda: open_scene_bands(product.name, product.s3_path),
    )
    print(
        f"images   : {done.from_memory} from the pixels read, {done.read} read again "
        f"(near the box edge), {done.failed} failed -> {root}"
    )


def run_model(args: argparse.Namespace, settings, box: BBox, product: SarProduct) -> int:
    """Detect with the exported point detector on VV+VH, under the same masks as CFAR."""
    from core.inference.runner import DETECTOR, PointDetection, PointDetectorRunner
    from scripts.build_dataset import read_pair

    if not args.model.is_file():
        print(f"model not found: {args.model}")
        return 1
    try:
        scene, window, bands = read_pair(product, box)
    except S3AccessError as exc:
        print(f"could not read the scene: {exc}")
        return 1
    except ValueError as exc:  # read_window refuses an empty window
        print(f"the requested box does not intersect the scene ({exc})")
        return 1
    infra, standing = platform_mask(box, product.acq_start.date(), scene, window)
    coast = None if args.no_coastline else coastline_mask(box, scene, window, args.coast_buffer_m)
    masked = (bands == 0).any(axis=0)
    for layer in (infra, coast):
        if layer is not None:
            masked |= layer
    runner = PointDetectorRunner(args.model)
    started = time.perf_counter()
    peaks = runner.detect(bands, masked)
    found: list[PointDetection] = []
    if peaks:
        lons, lats = scene.lonlat(
            [r + window.row_off for r, _, _ in peaks], [c + window.col_off for _, c, _ in peaks]
        )
        found = [
            PointDetection(float(lon), float(lat), score)
            for lon, lat, (_, _, score) in zip(lons, lats, peaks, strict=True)
            if box.contains(float(lon), float(lat))
        ]
    version = args.model.stem.removeprefix("point_detector_")
    print(
        f"detector : point detector {version} (threshold {runner.threshold:g}, "
        f"{runner.tile} px tiles) in {time.perf_counter() - started:.1f}s -> "
        f"{len(found)} vessels in the box; {standing} standing platforms masked"
    )
    if not args.no_store:
        written = store_detections(
            settings.database_url, product, box, found, "dv", version, detector=DETECTOR
        )
        print(f"database : {written} detections stored ({DETECTOR}, {version})")
        cut_found_chips(found, scene, window, bands, product, settings.data_dir / "chips")
    return 0


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
    parser.add_argument(
        "--scene",
        help="product name of an older pass to read (within --days) instead of the newest",
    )
    parser.add_argument("--pol", choices=("vv", "vh"), default="vh")
    parser.add_argument("--pfa", type=float, default=1e-6)
    parser.add_argument(
        "--no-store", action="store_true", help="do not write detections to the database"
    )
    parser.add_argument(
        "--coast-buffer-m",
        type=float,
        default=DEFAULT_BUFFER_M,
        help="seaward buffer masked around land",
    )
    parser.add_argument(
        "--no-coastline", action="store_true", help="skip the coastline mask, for comparison"
    )
    parser.add_argument(
        "--detector", choices=("cfar", "model"), default="cfar", help="CFAR or the point detector"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models") / "point_detector_kaggle-v2.onnx",
        help="exported ONNX point detector (fp32); default run 2, chosen against GFW",
    )
    args = parser.parse_args()
    if not 0 < args.size_km <= MAX_SIZE_KM:
        parser.error(f"--size-km must be in (0, {MAX_SIZE_KM}]")

    settings = load_settings()
    box = box_around(args.lon, args.lat, args.size_km)
    product = latest_product_covering(args.lon, args.lat, box, args.days, args.scene)
    if product is None:
        print(f"no readable Sentinel-1 scene covers {args.lat}, {args.lon} in {args.days} days")
        return 1
    print(f"scene    : {product.name}")
    print(f"acquired : {product.acq_start:%Y-%m-%d %H:%M} UTC ({product.orbit_direction})")
    if args.detector == "model":
        return run_model(args, settings, box, product)

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
    coast = None if args.no_coastline else coastline_mask(box, scene, window, args.coast_buffer_m)
    nodata = pixels == 0
    masked = nodata.copy()
    for layer in (infra, coast):
        if layer is not None:
            masked |= layer
    coast_text = (
        f"coastline {int(coast.sum())} px (buffer {args.coast_buffer_m:g} m)"
        if coast is not None
        else "no coastline mask, interior polygons only"
    )
    print(
        f"masking  : {standing} standing platforms, {infra.sum() if infra is not None else 0} px; "
        f"{coast_text}; nodata {int(nodata.sum())} px"
    )

    config = CfarConfig(pfa=args.pfa, satellite=product.satellite)
    version = detector_version(config)
    started = time.perf_counter()
    candidates = detect_candidates(pixels, config=config, mask=masked)
    located = georeference(candidates, scene, window_offset=(window.row_off, window.col_off))
    detect_s = time.perf_counter() - started

    if coast is None:
        land = PolygonLandMask()
        offshore = [c for c in located if not land.is_land(c.lon, c.lat)]
    else:
        offshore = located  # land was masked before CFAR
    # On a rotated scene the pixel window is a little larger than the box. Keep
    # only detections inside the box, so every output covers exactly the area
    # asked for and a re-run replaces its stored rows instead of adding to them.
    found = [c for c in offshore if box.contains(c.lon, c.lat)]
    # Nodata and masked platforms were never searched. Their share of the box
    # is estimated from their share of the window.
    searched_km2 = args.size_km**2 * float(np.mean(~masked))
    print(
        f"detector : CFAR Pfa {args.pfa:g} (alpha {config.alpha:.2f}) in {detect_s:.1f}s -> "
        f"{len(found)} candidates in the box ({len(located) - len(offshore)} inland dropped, "
        f"{len(offshore) - len(found)} outside the box), "
        f"{len(found) / max(searched_km2, 1e-6) * 100:.1f} per 100 km2 searched"
    )
    print()
    print_table(found)

    stem = f"{product.name.removesuffix('.SAFE')}_{args.pol}_{args.lat:.3f}_{args.lon:.3f}"
    png = QUICKLOOK_DIR / f"{stem}.png"
    geojson = DETECTION_DIR / f"{stem}.geojson"
    write_png(png, render_preview(pixels, found, window, masked))
    write_geojson(geojson, product, found, version)
    print()
    print(f"preview  : {png}")
    print(f"geojson  : {geojson}")
    if not args.no_store:
        written = store_detections(settings.database_url, product, box, found, args.pol, version)
        print(f"database : {written} detections stored ({args.pol.upper()}, {version})")
    if infra is not None:
        print(PLATFORM_ATTRIBUTION)
    if coast is not None:
        print(COASTLINE_ATTRIBUTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
