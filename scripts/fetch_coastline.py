"""Build the coastline land mask from OpenStreetMap land polygons.

Downloads land-polygons-split-4326.zip (about 926 MB) unless it is already
present, extracts the shapefile next to it, keeps the polygons around the Gulf
and the Gulf of Oman, writes them to data/reference as a small numpy file,
and deletes the download and the extracted shapefile.

Source: https://osmdata.openstreetmap.de/data/land-polygons.html
        (c) OpenStreetMap contributors, ODbL 1.0

Run:  uv run --with pyshp python scripts/fetch_coastline.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.db.spatial import BBox  # noqa: E402
from core.net import USER_AGENT, open_with_retry  # noqa: E402
from core.sar.coastline import (  # noqa: E402
    ATTRIBUTION,
    DEFAULT_PATH,
    LandPolygon,
    load_land,
    save_polygons,
)

SOURCE_URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"
ZIP_PATH = ROOT / "data" / "reference" / "land-polygons-split-4326.zip"
WORK_DIR = ROOT / "data" / "reference" / "_land_polygons_work"
MEMBER_PREFIX = "land-polygons-split-4326/land_polygons."
SHAPEFILE_PARTS = ("shp", "shx", "dbf", "prj", "cpg")

# The union of every catalogued scene footprint (44.84-60.73 E, 22.21-32.20 N
# on 2026-09-11), rounded outward.
EXTRACT_BBOX = BBox(44.5, 22.0, 61.0, 32.5)

CHUNK_BYTES = 1 << 20


def download(url: str, destination: Path) -> None:
    """Stream to a partial file, resuming if one is left over, then rename."""
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with open_with_retry(request, timeout=300) as response:
        # A server that ignores Range sends the whole file again.
        mode = "ab" if offset and response.status == 206 else "wb"
        with partial.open(mode) as out:
            shutil.copyfileobj(response, out, CHUNK_BYTES)
    partial.replace(destination)


def extract_shapefile(zip_path: Path, work_dir: Path) -> Path:
    """Extract the shapefile parts to disk; reading 1.3 GB through the zip is slower."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for part in SHAPEFILE_PARTS:
            member = MEMBER_PREFIX + part
            target = work_dir / f"land_polygons.{part}"
            with archive.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, CHUNK_BYTES)
    return work_dir / "land_polygons.shp"


def polygons_in(shapefile_path: Path, bbox: BBox) -> list[LandPolygon]:
    """Every land polygon whose extent touches ``bbox``, holes kept with their exterior."""
    import shapefile

    polygons: list[LandPolygon] = []
    extent = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    with shapefile.Reader(str(shapefile_path)) as reader:
        for shape in reader.iterShapes(bbox=extent):
            if shape is None:
                continue
            geometry = shape.__geo_interface__
            if geometry["type"] == "Polygon":
                parts = [geometry["coordinates"]]
            elif geometry["type"] == "MultiPolygon":
                parts = geometry["coordinates"]
            else:
                continue
            for exterior, *holes in parts:
                polygons.append(
                    LandPolygon(
                        np.asarray(exterior, dtype=np.float64),
                        holes=tuple(np.asarray(hole, dtype=np.float64) for hole in holes),
                    )
                )
    return polygons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument(
        "--keep-download", action="store_true", help="keep the zip and extracted shapefile"
    )
    parser.add_argument("--path", type=Path, default=ROOT / DEFAULT_PATH)
    args = parser.parse_args()

    if args.path.is_file() and not args.force:
        print(f"{args.path} already present; use --force to rebuild")
    else:
        if not ZIP_PATH.is_file():
            print(f"downloading {SOURCE_URL} (about 926 MB)...", flush=True)
            download(SOURCE_URL, ZIP_PATH)
        print(f"extracting shapefile from {ZIP_PATH.name}...", flush=True)
        started = time.perf_counter()
        shapefile_path = extract_shapefile(ZIP_PATH, WORK_DIR)
        print(f"reading polygons touching {EXTRACT_BBOX}...", flush=True)
        polygons = polygons_in(shapefile_path, EXTRACT_BBOX)
        save_polygons(args.path, polygons, source=f"{SOURCE_URL} bbox={EXTRACT_BBOX}")
        print(f"built in {time.perf_counter() - started:.0f}s")
        if not args.keep_download:
            shutil.rmtree(WORK_DIR)
            ZIP_PATH.unlink()
            print("removed the download and the extracted shapefile")

    loaded = load_land(args.path)
    rings = sum(1 + len(p.holes) for p in loaded)
    vertices = sum(len(p.exterior) + sum(len(h) for h in p.holes) for p in loaded)
    print()
    print(f"land polygons : {len(loaded)}")
    print(f"rings         : {rings}")
    print(f"vertices      : {vertices}")
    print(f"file          : {args.path} ({args.path.stat().st_size / 1e6:.1f} MB)")
    print(ATTRIBUTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
