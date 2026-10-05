"""Phase 6 v1, product path: DARK_VESSEL events for one stored pass.

Reads our stored detections of one pass inside a box (per pass the model's
where it ran, CFAR's otherwise), fetches GFW hourly AIS presence for H-1..H+1
over the box buffered by 25 km (cached under data/fusion/presence) and GFW's
SAR detections for the pass, fuses them at the operating margin chosen on
validation passes (models/runs/kaggle-v2/fusion_v1/summary.json, or
--margin), and stores the dark-vessel events with their evidence cards,
replacing this run's earlier events for the pass. GFW publishes presence
about 96 h and SAR detections about 5 days after a pass, so younger passes
are refused. Passes whose presence is missing or evidently incomplete give no
events (core.fusion.events). One GFW report at a time; never alongside
another GFW job.

    uv run --offline --python 3.12 --no-project --with numpy --with scipy --with rasterio \
        --with boto3 --with sqlalchemy --with httpx --with gfw-api-python-client \
        python -u scripts/fuse_scene.py --lon 56.53 --lat 25.22 --size-km 20 [--scene NAME]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db.repository import DetectionView  # noqa: E402

# GFW SAR detections are published about 5 days after a pass, presence about 96 h.
MIN_PASS_AGE = dt.timedelta(days=5)
RUN = "fusion-v1"
# Model detections are used only from the product model (run 2); older runs'
# rows may still be stored (the 2026-09-27 Fujairah pass holds run 1's).
PRODUCT_MODEL_VERSION = "kaggle-v2"
MODEL_DETECTOR = "pointdet"
SUMMARY = Path("models/runs/kaggle-v2/fusion_v1/summary.json")
PRESENCE_CACHE = Path("data/fusion")


@dataclass(frozen=True)
class StoredPass:
    scene_id: int
    product_name: str
    acquired: dt.datetime
    detections: list[DetectionView]


def pick_pass(
    views: Sequence[DetectionView], *, now: dt.datetime, scene: str | None = None
) -> StoredPass:
    """The named pass, or the newest stored pass old enough for GFW to have published its data.

    Model detections from other than the product model's version are ignored.
    """
    by_scene: dict[int, list[DetectionView]] = {}
    for v in views:
        if v.detector == MODEL_DETECTOR and v.detector_version != PRODUCT_MODEL_VERSION:
            continue
        by_scene.setdefault(v.scene_id, []).append(v)
    passes = [
        StoredPass(sid, vs[0].product_name, vs[0].acq_time, vs) for sid, vs in by_scene.items()
    ]
    if scene is not None:
        named = [p for p in passes if p.product_name == scene]
        if not named:
            raise ValueError(f"no stored detections of {scene} in the box")
        if now - named[0].acquired < MIN_PASS_AGE:
            raise ValueError(
                f"{scene} is younger than {MIN_PASS_AGE.days} days; GFW has not published it"
            )
        return named[0]
    ready = [p for p in passes if now - p.acquired >= MIN_PASS_AGE]
    if not ready:
        raise ValueError(f"no stored pass in the box is at least {MIN_PASS_AGE.days} days old")
    return max(ready, key=lambda p: p.acquired)


def operating_margin(path: Path) -> float:
    """The margin chosen on validation passes by scripts/fuse_passes.py --summary."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run scripts/fuse_passes.py --summary, or pass --margin"
        )
    return float(json.loads(path.read_text(encoding="utf-8"))["operating_margin_m"])


async def _sar_refs(token: str, box, acquired: dt.datetime) -> list[tuple]:
    import gfwapiclient

    from core.labeling.gfw import cells_for_pass
    from core.labeling.gfw_api import fetch_sar_cells

    client = gfwapiclient.Client(access_token=token)
    day = acquired.date()
    cells = await fetch_sar_cells(client, box, day, day + dt.timedelta(days=1))
    return [(c.lon, c.lat, c.detections, c.matched) for c in cells_for_pass(cells, acquired)]


def main(argv: Sequence[str] | None = None) -> int:
    from core.config import load_settings
    from core.db.repository import DetectionRepository, EventRepository
    from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
    from core.fusion.events import dark_vessel_events
    from core.fusion.geometry import buffer_box
    from core.fusion.prefilter import SarDetection
    from core.sar.coastline import load_land
    from core.sar.infrastructure import InfrastructureMask, load_platforms
    from scripts.detect_window import box_around
    from scripts.fuse_passes import (
        EDGE_M,
        INFRASTRUCTURE_RADIUS_M,
        LAND_REACH_M,
        MARGINS_M,
        R_OVER_V_BOUND_S,
        SHORE_M,
        presence_for,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--size-km", type=float, default=20.0)
    parser.add_argument(
        "--scene", help="product name of the stored pass (default: newest old enough)"
    )
    parser.add_argument(
        "--margin", type=float, help="gate margin in m (default: chosen on validation)"
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        print("GFW_API_TOKEN is not set (.env)", flush=True)
        return 1
    margin = args.margin if args.margin is not None else operating_margin(SUMMARY)
    box = box_around(args.lon, args.lat, args.size_km)
    engine = create_db_engine(settings.database_url)
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        views = DetectionRepository(session).in_bbox(box, detector="auto", limit=100_000)
    stored = pick_pass(views, now=dt.datetime.now(dt.UTC), scene=args.scene)
    result_like = {
        "area": "window",
        "scene": stored.product_name,
        "acquired": stored.acquired.isoformat(),
    }
    fixes = presence_for(PRESENCE_CACHE, result_like, box, token)
    refs = asyncio.run(_sar_refs(token, box, stored.acquired))
    events, summary = dark_vessel_events(
        scene=stored.product_name,
        acquired=stored.acquired,
        detections=[
            (
                v.id,
                SarDetection(
                    lon=v.lon,
                    lat=v.lat,
                    score=v.confidence,
                    detector=v.detector,
                    detector_version=v.detector_version,
                ),
            )
            for v in stored.detections
        ],
        fixes=fixes,
        refs=refs,
        land=load_land(bbox=buffer_box(box, LAND_REACH_M)),
        infrastructure=InfrastructureMask(
            load_platforms(bbox=buffer_box(box, INFRASTRUCTURE_RADIUS_M + 500.0)),
            radius_m=INFRASTRUCTURE_RADIUS_M,
        ).for_date(stored.acquired.date()),
        box=box,
        margin_m=margin,
        margins=MARGINS_M,
        r_over_v_s=R_OVER_V_BOUND_S,
        shore_m=SHORE_M,
        edge_m=EDGE_M,
    )
    with session_scope(factory) as session:
        written = EventRepository(session).replace_for_scene(stored.scene_id, RUN, events)
    detectors = sorted({v.detector for v in stored.detections})
    print(
        f"{stored.product_name} ({stored.acquired:%Y-%m-%d %H:%M} UTC, {'/'.join(detectors)}), margin {margin:g} m"
    )
    print(f"  {json.dumps(summary)}")
    print(
        f"  {written} DARK_VESSEL events stored (run {RUN}). Powered by Global Fishing Watch.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
