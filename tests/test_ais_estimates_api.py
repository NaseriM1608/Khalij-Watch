"""Observed coordinates and experimental estimates must remain distinct."""

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.ais.messages import AisPositionReport
from core.config import Settings, load_settings
from core.db.repository import PositionRepository, VesselRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import GULF_BBOX, HORMUZ_BBOX
from services.api import main as api


@pytest.fixture
def live_client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    now = dt.datetime.now(dt.UTC)
    with session_scope(factory) as session:
        VesselRepository(session).ensure_exists({422123456}, seen_at=now)
        session.flush()
        PositionRepository(session).add_batch(
            [
                AisPositionReport(
                    mmsi=422123456,
                    ts=now - dt.timedelta(seconds=60),
                    lon=56.5,
                    lat=26.5,
                    source="openwaters",
                    sog_kn=12,
                    cog_deg=90,
                    timestamp_basis="provider",
                    received_at=now,
                    provider_source="aishub",
                    provider_reencoded=True,
                ),
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
    with TestClient(api.app) as client:
        api._state["sessions"] = factory
        api._state["settings"] = settings
        yield client, factory
    api._state.clear()
    engine.dispose()


def test_estimate_does_not_replace_observation(live_client):
    client, _ = live_client
    vessel = client.get("/api/vessels").json()["vessels"][0]
    assert (vessel["lon"], vessel["lat"]) == (56.5, 26.5)
    assert vessel["estimate"]["status"] == "estimated"
    assert vessel["estimate"]["lon"] > vessel["lon"]
    assert vessel["timestamp_basis"] == "provider"
    assert vessel["provider_reencoded"] is True


def test_estimates_can_be_disabled_for_radar_history(live_client):
    client, _ = live_client
    estimate = client.get("/api/vessels?estimates=false").json()["vessels"][0]["estimate"]
    assert estimate["status"] == "unavailable"
    assert estimate["reason"] == "disabled"


def test_health_means_observations_not_credentials(live_client):
    client, _ = live_client
    health = client.get("/api/health").json()
    assert health["has_live_ais"] is True  # no credentials, but observed data arrived
    assert health["fresh_ais_vessels"] == 1
    assert health["ais_status"] == "fresh"


def test_old_messages_do_not_become_fresh_on_receipt(live_client):
    from sqlalchemy import update

    from core.db.models import AisPosition

    client, factory = live_client
    with session_scope(factory) as session:
        session.execute(
            update(AisPosition).values(ts=dt.datetime.now(dt.UTC) - dt.timedelta(hours=1))
        )
    health = client.get("/api/health").json()
    assert health["has_live_ais"] is False
    assert health["ais_status"] == "stale"
    vessel = client.get("/api/vessels").json()["vessels"][0]
    assert vessel["estimate"]["reason"] == "stale"


def test_openwaters_is_explicitly_enabled_without_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENWATERS_ENABLED", "1")
    settings = load_settings(tmp_path / "missing.env")
    assert settings.openwaters_enabled is True
    assert "openwaters" in settings.describe_feeds()
