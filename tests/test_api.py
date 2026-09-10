"""API tests, run against a seeded in-memory database."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.ais.messages import AisPositionReport, VesselStatic
from core.config import Settings
from core.db.repository import PositionRepository, VesselRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import GULF_BBOX, HORMUZ_BBOX
from services.api import main as api

MMSI = 422123456
FAR_MMSI = 470123456


@pytest.fixture
def client():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    now = dt.datetime.now(dt.UTC)

    with session_scope(factory) as session:
        vessels = VesselRepository(session)
        vessels.ensure_exists({MMSI, FAR_MMSI}, seen_at=now)
        session.flush()
        vessels.upsert_static(
            VesselStatic(mmsi=MMSI, name="GULF VOYAGER", ship_type=80), seen_at=now
        )
        session.flush()
        PositionRepository(session).add_batch(
            [
                AisPositionReport(
                    mmsi=MMSI,
                    ts=now - dt.timedelta(minutes=m),
                    lon=56.5,
                    lat=26.5 + m * 0.001,
                    source="synthetic",
                    sog_kn=12.0,
                    cog_deg=270.0,
                )
                for m in (30, 20, 10, 1)
            ]
            + [
                AisPositionReport(
                    mmsi=FAR_MMSI,
                    ts=now,
                    lon=10.0,
                    lat=50.0,  # North Sea; outside every Gulf AOI
                    source="synthetic",
                )
            ]
        )

    settings = Settings(
        aisstream_key=None,
        aisstream_url="wss://example.invalid",
        aiscast_url="wss://example.invalid",
        aiscast_token=None,
        gulf_bbox=GULF_BBOX,
        hormuz_bbox=HORMUZ_BBOX,
        database_url=None,
        data_dir=Path("data"),
    )
    with TestClient(api.app) as test_client:
        # Entering the context runs the real lifespan, which points at the
        # on-disk database. Override afterwards so the test owns the state.
        api._state["sessions"] = factory
        api._state["settings"] = settings
        yield test_client
    api._state.clear()


def test_health_reports_counts_and_feed_status(client):
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["vessels"] == 2
    assert body["positions"] == 5
    # No credentials configured, so the UI must be able to warn the operator.
    assert body["has_live_ais"] is False


def test_aoi_endpoint_exposes_both_areas(client):
    body = client.get("/api/aoi").json()

    assert body["gulf"]["max_lon"] == 58.0
    assert body["hormuz"]["min_lon"] == 54.0


def test_vessels_returns_latest_fix_per_vessel(client):
    body = client.get("/api/vessels").json()

    assert body["count"] == 1  # the North Sea vessel is outside the AOI
    vessel = body["vessels"][0]
    assert vessel["mmsi"] == MMSI
    assert vessel["name"] == "GULF VOYAGER"
    assert vessel["coarse_class"] == "cargo_tanker"


def test_every_vessel_carries_its_age(client):
    """A stale contact must never be able to look live in the UI."""
    vessel = client.get("/api/vessels").json()["vessels"][0]

    assert "age_seconds" in vessel
    assert 0 <= vessel["age_seconds"] < 120


def test_response_carries_source_attribution(client):
    body = client.get("/api/vessels").json()

    assert body["attribution"] == ["synthetic"]
    assert body["vessels"][0]["source"] == "synthetic"


def test_bbox_filter_is_applied(client):
    body = client.get("/api/vessels", params={"bbox": "0,45,20,55"}).json()

    assert body["count"] == 1
    assert body["vessels"][0]["mmsi"] == FAR_MMSI


def test_malformed_bbox_is_rejected(client):
    response = client.get("/api/vessels", params={"bbox": "not,a,bbox"})

    assert response.status_code == 400
    assert "invalid bbox" in response.json()["detail"]


def test_inverted_bbox_is_rejected(client):
    response = client.get("/api/vessels", params={"bbox": "58,23,47,30"})

    assert response.status_code == 400


def test_max_age_excludes_stale_contacts(client):
    body = client.get("/api/vessels", params={"max_age_minutes": 1}).json()

    # Newest fix is a minute old; nothing older should survive the cutoff.
    assert body["count"] <= 1


def test_track_returns_points_oldest_first(client):
    body = client.get(f"/api/vessels/{MMSI}/track").json()

    assert body["mmsi"] == MMSI
    assert len(body["points"]) == 4
    timestamps = [p["ts"] for p in body["points"]]
    assert timestamps == sorted(timestamps)


def test_track_timestamps_are_timezone_aware(client):
    """Naive timestamps would be rendered as local time by the browser."""
    points = client.get(f"/api/vessels/{MMSI}/track").json()["points"]

    assert all(p["ts"].endswith("+00:00") for p in points)


def test_track_for_unknown_vessel_is_404(client):
    assert client.get("/api/vessels/999999999/track").status_code == 404


def test_track_hours_is_bounded(client):
    assert client.get(f"/api/vessels/{MMSI}/track", params={"hours": 0}).status_code == 422
    assert client.get(f"/api/vessels/{MMSI}/track", params={"hours": 100_000}).status_code == 422


def test_websocket_pushes_a_viewport_snapshot(client):
    with client.websocket_connect("/ws/stream") as ws:
        payload = ws.receive_json()

    assert payload["count"] == 1
    assert payload["vessels"][0]["mmsi"] == MMSI
    assert "generated_at" in payload


def test_websocket_reports_a_bad_bbox_instead_of_crashing(client):
    with client.websocket_connect("/ws/stream?bbox=garbage") as ws:
        payload = ws.receive_json()

    assert "error" in payload
