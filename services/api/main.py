"""HTTP and WebSocket API for the live map.

Run:  uv run uvicorn services.api.main:app --reload

Design notes
------------
* Every vessel returned carries ``age_seconds`` explicitly. The UI must be able
  to show how old a contact is; a stale contact must never look live.
* Every vessel carries ``source``. Feed licences require attribution to be
  carried through to the display.
* Browsers must never connect to aisstream directly - the provider forbids it -
  so the frontend talks only to this API.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.config import load_settings
from core.db.repository import (
    DetectionRepository,
    DetectionView,
    PositionRepository,
    SceneRepository,
    VesselFix,
    VesselRepository,
)
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from core.sar.catalogue import parse_polygon_wkt

logger = logging.getLogger(__name__)

# How often the websocket pushes a refreshed viewport.
STREAM_INTERVAL_S = 5.0
MAX_TRACK_HOURS = 24 * 7

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = load_settings()
    engine = create_db_engine(settings.database_url)
    create_all(engine)
    _state["settings"] = settings
    _state["sessions"] = make_session_factory(engine)
    logger.info("API ready; AOI=%s", settings.gulf_bbox)
    yield
    _state.clear()


app = FastAPI(title="khalij-watch", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _sessions():
    factory = _state.get("sessions")
    if factory is None:
        raise HTTPException(status_code=503, detail="service starting")
    return factory


def _parse_bbox(spec: str | None) -> BBox:
    settings = _state.get("settings")
    if not spec:
        return settings.gulf_bbox if settings else BBox(47.5, 23.5, 58.0, 30.5)
    try:
        return BBox.parse(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid bbox: {exc}") from exc


def _fix_to_json(fix: VesselFix, now: dt.datetime) -> dict[str, Any]:
    return {
        "mmsi": fix.mmsi,
        "lon": round(fix.lon, 6),
        "lat": round(fix.lat, 6),
        "ts": fix.ts.isoformat(),
        # Explicit, so the UI never has to imply freshness it cannot verify.
        "age_seconds": round(fix.age_seconds(now)),
        "sog_kn": fix.sog_kn,
        "cog_deg": fix.cog_deg,
        "heading_deg": fix.heading_deg,
        "name": fix.name,
        "coarse_class": fix.coarse_class,
        "source": fix.source,
        "plausibility_score": fix.plausibility_score,
        # AIS in this AOI is adversarial; the UI must be able to show doubt.
        "is_suspect": fix.is_suspect,
        "anomaly_flags": fix.anomaly_flags,
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    settings = _state.get("settings")
    with session_scope(_sessions()) as session:
        vessels = VesselRepository(session).count()
        positions = PositionRepository(session).count()
        scenes_known = SceneRepository(session).count()
        detections_stored = DetectionRepository(session).count()
    return {
        "status": "ok",
        "vessels": vessels,
        "positions": positions,
        "sar_scenes": scenes_known,
        "detections": detections_stored,
        "live_feeds": settings.describe_feeds() if settings else [],
        # Surfaced so the UI can warn when it is showing seeded, not live, data.
        "has_live_ais": bool(settings and settings.has_live_ais),
    }


@app.get("/api/aoi")
async def aoi() -> dict[str, Any]:
    settings = _state["settings"]
    return {
        "gulf": _bbox_json(settings.gulf_bbox),
        "hormuz": _bbox_json(settings.hormuz_bbox),
    }


def _bbox_json(bbox: BBox) -> dict[str, float]:
    return {
        "min_lon": bbox.min_lon,
        "min_lat": bbox.min_lat,
        "max_lon": bbox.max_lon,
        "max_lat": bbox.max_lat,
    }


@app.get("/api/vessels")
async def vessels(
    bbox: str | None = Query(default=None, description="min_lon,min_lat,max_lon,max_lat"),
    max_age_minutes: int = Query(default=360, ge=1, le=60 * 24 * 30),
    limit: int = Query(default=10_000, ge=1, le=50_000),
) -> dict[str, Any]:
    """Latest known position per vessel inside a bounding box."""
    box = _parse_bbox(bbox)
    now = dt.datetime.now(dt.UTC)
    with session_scope(_sessions()) as session:
        fixes = PositionRepository(session).latest_in_bbox(
            box,
            now=now,
            max_age=dt.timedelta(minutes=max_age_minutes),
            limit=limit,
        )
    return {
        "generated_at": now.isoformat(),
        "bbox": _bbox_json(box),
        "count": len(fixes),
        "vessels": [_fix_to_json(f, now) for f in fixes],
        "suspect_count": sum(1 for f in fixes if f.is_suspect),
        "attribution": sorted({f.source for f in fixes}),
    }


@app.get("/api/vessels/{mmsi}/track")
async def track(
    mmsi: int, hours: int = Query(default=24, ge=1, le=MAX_TRACK_HOURS)
) -> dict[str, Any]:
    """One vessel's recent trail, oldest point first."""
    now = dt.datetime.now(dt.UTC)
    since = now - dt.timedelta(hours=hours)
    with session_scope(_sessions()) as session:
        vessel = VesselRepository(session).get(mmsi)
        if vessel is None:
            raise HTTPException(status_code=404, detail=f"unknown MMSI {mmsi}")
        points = PositionRepository(session).track(mmsi, since=since)
        payload = {
            "mmsi": mmsi,
            "name": vessel.name,
            "coarse_class": vessel.coarse_class,
            "imo": vessel.imo,
            "flag": vessel.flag,
            "sanctioned": vessel.sanctioned,
            "points": [
                {
                    "lon": p.lon,
                    "lat": p.lat,
                    "ts": (
                        p.ts.isoformat() if p.ts.tzinfo else p.ts.replace(tzinfo=dt.UTC).isoformat()
                    ),
                    "sog_kn": p.sog_kn,
                    "source": p.source,
                }
                for p in points
            ],
        }
    return payload


@app.get("/api/scenes")
async def scenes(
    bbox: str | None = Query(default=None),
    days: int = Query(default=14, ge=1, le=365),
    limit: int = Query(default=300, ge=1, le=2000),
) -> dict[str, Any]:
    """Radar scenes covering an area, newest first.

    This is the coverage-honesty layer. Where no recent scene exists the
    platform is blind, and the UI must be able to say so rather than implying
    the absence of detections means the absence of vessels.
    """
    box = _parse_bbox(bbox)
    now = dt.datetime.now(dt.UTC)
    since = now - dt.timedelta(days=days)

    with session_scope(_sessions()) as session:
        found = SceneRepository(session).intersecting(box, since=since, limit=limit)
        payload = [
            {
                "id": scene.id,
                "product_name": scene.product_name,
                "satellite": scene.satellite,
                "polarisation": scene.polarisation,
                "orbit_direction": scene.orbit_direction,
                "acq_time": _iso_utc(scene.acq_time),
                "age_hours": round((now - _utc(scene.acq_time)).total_seconds() / 3600, 1),
                "status": scene.status,
                "footprint": parse_polygon_wkt(scene.footprint_wkt),
                "bbox": {
                    "min_lon": scene.min_lon,
                    "min_lat": scene.min_lat,
                    "max_lon": scene.max_lon,
                    "max_lat": scene.max_lat,
                },
            }
            for scene in found
        ]

    newest = max((s["acq_time"] for s in payload), default=None)
    return {
        "generated_at": now.isoformat(),
        "count": len(payload),
        "scenes": payload,
        "last_look": newest,
        "satellites": sorted({s["satellite"] for s in payload}),
        "attribution": "Contains modified Copernicus Sentinel data",
    }


@app.get("/api/detections")
async def detections(
    bbox: str | None = Query(default=None, description="min_lon,min_lat,max_lon,max_lat"),
    days: int = Query(default=14, ge=1, le=365),
    limit: int = Query(default=20_000, ge=1, le=100_000),
    detector: Literal["auto", "cfar", "pointdet", "all"] = Query(default="auto"),
) -> dict[str, Any]:
    """Radar detections from recent passes, newest pass first.

    Each carries the age of the pass that saw it. A detection is a snapshot of
    one moment, hours or days old, and the UI must never draw it as a live
    contact. By default each pass shows one detector's results - the model's
    where it ran, CFAR's elsewhere - so no vessel is drawn twice.
    """
    box = _parse_bbox(bbox)
    now = dt.datetime.now(dt.UTC)
    since = now - dt.timedelta(days=days)
    with session_scope(_sessions()) as session:
        found = DetectionRepository(session).in_bbox(
            box, since=since, limit=limit, detector=detector
        )
    return {
        "generated_at": now.isoformat(),
        "bbox": _bbox_json(box),
        "detector": detector,
        "count": len(found),
        "detections": [_detection_to_json(d, now) for d in found],
        "attribution": "Contains modified Copernicus Sentinel data",
    }


def _detection_to_json(detection: DetectionView, now: dt.datetime) -> dict[str, Any]:
    return {
        "id": detection.id,
        "scene_id": detection.scene_id,
        "lon": round(detection.lon, 6),
        "lat": round(detection.lat, 6),
        "acq_time": detection.acq_time.isoformat(),
        # Explicit, so a detection from an old pass can never look current.
        "age_hours": round(detection.age_seconds(now) / 3600, 2),
        "satellite": detection.satellite,
        "polarisation": detection.polarisation,
        "length_m": _rounded(detection.length_m),
        "width_m": _rounded(detection.width_m),
        # Modulo 180: radar cannot tell bow from stern.
        "heading_deg": _rounded(detection.heading_deg),
        "confidence": detection.confidence,
        "detector": detection.detector,
        "detector_version": detection.detector_version,
        "pixel_count": detection.pixel_count,
        "peak_dn": _rounded(detection.peak_dn),
        "product_name": detection.product_name,
    }


def _rounded(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


def _utc(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo else value.replace(tzinfo=dt.UTC)


def _iso_utc(value: dt.datetime) -> str:
    return _utc(value).isoformat()


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket, bbox: str | None = None) -> None:
    """Push refreshed viewport contents on an interval.

    At Gulf densities (a few thousand active vessels) sending GeoJSON-ish JSON
    is comfortably cheap. Vector tiles are premature below ~50k objects.
    """
    await websocket.accept()
    try:
        box = _parse_bbox(bbox)
    except HTTPException as exc:
        await websocket.send_json({"error": exc.detail})
        await websocket.close()
        return

    try:
        while True:
            now = dt.datetime.now(dt.UTC)
            with session_scope(_sessions()) as session:
                fixes = PositionRepository(session).latest_in_bbox(box, now=now)
            await websocket.send_json(
                {
                    "generated_at": now.isoformat(),
                    "count": len(fixes),
                    "vessels": [_fix_to_json(f, now) for f in fixes],
                }
            )
            await asyncio.sleep(STREAM_INTERVAL_S)
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("stream socket failed")
        await websocket.close()
