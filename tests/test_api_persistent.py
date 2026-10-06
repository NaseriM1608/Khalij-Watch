"""Tests for marking detections and dark-vessel candidates at persistent spots.

Something detected at the same spot pass after pass is likely a fixed object
(a jetty, mooring, buoy or unmapped platform) or a ship moored for months
(core/persistence.py). The console marks such detections; it hides nothing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from core.areas import box_around
from core.config import Settings
from core.db.repository import DetectionRepository, EventRepository, NewEvent, SceneRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from core.persistence import Spot, save_spots
from services.api import main as api

NOW = dt.datetime.now(dt.UTC)
SCENE = "S1D_IW_GRDH_1SDV_20260930T023834_TEST_COG.SAFE"
SPOT = Spot(50.30, 29.25, "kharg", 9, 12, "2024-01-05", "2025-11-03")
AT_SPOT, ELSEWHERE = (50.3005, 29.2504), (50.35, 29.30)


@dataclass(frozen=True)
class Extent:
    min_lon: float = 49.0
    min_lat: float = 28.0
    max_lon: float = 52.0
    max_lat: float = 30.5


@dataclass(frozen=True)
class Product:
    product_id: str = SCENE
    name: str = SCENE
    acq_start: dt.datetime = NOW - dt.timedelta(days=6)
    satellite: str = "S1D"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "DESCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
    footprint: tuple = ((49.0, 28.0), (52.0, 28.0), (52.0, 30.5), (49.0, 30.5), (49.0, 28.0))
    bbox: Extent = Extent()


@dataclass(frozen=True)
class Peak:
    lon: float
    lat: float
    confidence: float = 0.9
    length_m: None = None
    width_m: None = None
    heading_deg: None = None
    pixel_count: None = None
    peak_amplitude: None = None


@pytest.fixture
def client(tmp_path):
    (tmp_path / "reference").mkdir()
    save_spots(tmp_path / "reference" / "persistent_spots.json", [SPOT], {"before": "2026-07-01"})
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        SceneRepository(session).upsert(Product())
        scene_id = SceneRepository(session).id_for(SCENE)
        DetectionRepository(session).replace_in_bbox(
            scene_id,
            box_around(50.30, 29.25, 20.0),
            [Peak(*AT_SPOT), Peak(*ELSEWHERE)],
            polarisation="DV",
            detector="pointdet",
            detector_version="kaggle-v2",
            confidence=1.0,
            shore_m=[2000.0, 2000.0],
        )
        EventRepository(session).replace_for_scene(
            scene_id,
            "fusion-v1",
            [
                NewEvent(
                    event_type="DARK_VESSEL",
                    ts=NOW - dt.timedelta(days=6),
                    lon=AT_SPOT[0],
                    lat=AT_SPOT[1],
                    severity="high",
                    detection_id=None,
                    evidence={"status": "dark"},
                )
            ],
        )
    settings = Settings(
        aisstream_key=None,
        aisstream_url="wss://example.invalid",
        aiscast_url="wss://example.invalid",
        aiscast_token=None,
        gulf_bbox=BBox(47.5, 23.5, 58.0, 30.5),
        hormuz_bbox=BBox(54.0, 25.0, 58.0, 27.5),
        database_url=None,
        data_dir=tmp_path,
    )
    with TestClient(api.app) as test_client:
        api._state["sessions"] = factory
        api._state["settings"] = settings
        yield test_client
    api._state.clear()


EXPECTED = {
    "area": "kharg",
    "passes_seen": 9,
    "passes_imaged": 12,
    "first": "2024-01-05",
    "last": "2025-11-03",
}


def test_a_detection_at_a_persistent_spot_is_marked_and_others_are_not(client):
    found = client.get("/api/detections", params={"detector": "product", "days": 30}).json()
    by_position = {(d["lon"], d["lat"]): d for d in found["detections"]}
    assert by_position[AT_SPOT]["persistent"] == EXPECTED
    assert by_position[ELSEWHERE]["persistent"] is None


def test_a_dark_vessel_candidate_at_a_persistent_spot_is_marked_not_hidden(client):
    (event,) = client.get("/api/events", params={"type": "DARK_VESSEL"}).json()["events"]
    assert event["persistent"] == EXPECTED


def test_without_a_spots_file_nothing_is_marked(client, tmp_path):
    (tmp_path / "reference" / "persistent_spots.json").unlink()
    found = client.get("/api/detections", params={"detector": "product", "days": 30}).json()
    assert {d["persistent"] for d in found["detections"]} == {None}
