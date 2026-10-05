"""Tests for /api/events: dark-vessel alerts with their evidence cards.

Every event carries its evidence and its age: an alert from a pass days old
must never look live, and one that cannot be explained is never served
(PLAN §9, §10).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from core.db.repository import EventRepository, NewEvent, SceneRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from core.fusion.chips import chip_relpath, image_meta
from services.api import main as api

ACQ = dt.datetime(2026, 9, 25, 14, 15, 51, tzinfo=dt.UTC)
SCENE = "S1C_IW_GRDH_1SDV_20260925T141551_TEST_COG.SAFE"
CHIP_BYTES = b"\x89PNG\r\n\x1a\nchip"


@dataclass(frozen=True)
class Extent:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@dataclass(frozen=True)
class Product:
    product_id: str = "S1C_IW_GRDH_TEST"
    name: str = "S1C_IW_GRDH_TEST"
    acq_start: dt.datetime = ACQ
    satellite: str = "S1C"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "ASCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
    footprint: tuple = ((55.8, 24.0), (58.7, 24.0), (58.7, 26.0), (55.8, 26.0), (55.8, 24.0))
    bbox: Extent = Extent(55.8, 24.0, 58.7, 26.0)


def event(
    lon: float, lat: float, event_type: str = "DARK_VESSEL", chip_pols: list[str] | None = None
) -> NewEvent:
    chip = {"scene": SCENE, "lon": lon, "lat": lat}
    if chip_pols is not None:
        chip["image"] = image_meta(chip_pols)
    return NewEvent(
        event_type=event_type,
        ts=ACQ,
        lon=lon,
        lat=lat,
        severity="high",
        detection_id=None,
        evidence={"status": "dark", "confidence": "high", "nearest_ais": [], "chip": chip},
    )


@pytest.fixture
def client(tmp_path):
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        SceneRepository(session).upsert(Product())
        scene_id = SceneRepository(session).id_for(Product().product_id)
        EventRepository(session).replace_for_scene(
            scene_id,
            "fusion-v1",
            [
                event(56.50, 25.20, chip_pols=["vv"]),
                event(56.51, 25.21, "ZONE_ENTRY"),
                event(52.0, 27.0),
            ],
        )
    chip = tmp_path / "chips" / chip_relpath(SCENE, 56.50, 25.20, "vv")
    chip.parent.mkdir(parents=True)
    chip.write_bytes(CHIP_BYTES)
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


def test_events_come_with_evidence_age_and_attribution(client):
    body = client.get("/api/events", params={"bbox": "56.4,25.1,56.6,25.3"}).json()
    assert body["count"] == 2
    first = body["events"][0]
    assert first["evidence"]["status"] == "dark"
    assert first["ts"] == ACQ.isoformat()
    assert first["age_hours"] > 0
    assert (first["run"], first["severity"]) == ("fusion-v1", "high")
    assert "Powered by Global Fishing Watch" in body["attribution"]


def test_events_filter_by_type(client):
    body = client.get(
        "/api/events", params={"bbox": "56.4,25.1,56.6,25.3", "type": "DARK_VESSEL"}
    ).json()
    assert [e["event_type"] for e in body["events"]] == ["DARK_VESSEL"]


def test_events_default_to_the_gulf(client):
    assert client.get("/api/events").json()["count"] == 3


def test_a_bad_bbox_is_rejected(client):
    assert client.get("/api/events", params={"bbox": "nonsense"}).status_code == 400


def chip_url(client, lon: float, pol: str = "vv") -> str:
    found = client.get("/api/events", params={"bbox": "56.4,25.1,56.6,25.3"}).json()["events"]
    (match,) = [e for e in found if e["lon"] == lon]
    return f"/api/events/{match['id']}/chip/{pol}"


def test_an_events_radar_chip_is_served_from_disk(client):
    response = client.get(chip_url(client, 56.50))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == CHIP_BYTES
    # Re-fusing a pass can reuse an event id for another detection: always revalidate.
    assert response.headers["cache-control"] == "no-cache"


def test_the_card_says_which_chips_exist(client):
    found = client.get("/api/events", params={"bbox": "56.4,25.1,56.6,25.3"}).json()["events"]
    images = {e["lon"]: e["evidence"]["chip"].get("image") for e in found}
    assert images[56.50]["pols"] == ["vv"]
    assert images[56.51] is None


def test_a_chip_not_cut_is_not_found(client):
    assert client.get(chip_url(client, 56.50, "vh")).status_code == 404
    assert client.get(chip_url(client, 56.51)).status_code == 404


def test_an_unknown_event_or_polarisation_has_no_chip(client):
    assert client.get("/api/events/999999/chip/vv").status_code == 404
    assert client.get(chip_url(client, 56.50, "hh")).status_code == 422
    assert client.get(f"/api/events/{2**64}/chip/vv").status_code == 422
    assert client.get("/api/events/0/chip/vv").status_code == 422
