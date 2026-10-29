"""API tests, run against a seeded in-memory database."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.ais.messages import AisPositionReport, VesselStatic
from core.config import Settings
from core.db.repository import PositionRepository, VesselRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import GULF_BBOX, HORMUZ_BBOX, BBox
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


# --------------------------------------------------------------------------
# SAR coverage
# --------------------------------------------------------------------------


def _seed_scene(factory, product_id="s1", lon0=56.0, lat0=26.0, hours_ago=3):
    from core.db.repository import SceneRepository

    class FakeProduct:
        pass

    product = FakeProduct()
    product.product_id = product_id
    product.name = f"S1D_IW_GRDH_1SDV_{product_id}_COG.SAFE"
    product.satellite = "S1D"
    product.mode = "IW"
    product.polarisation = "DV"
    product.acq_start = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours_ago)
    product.orbit_direction = "DESCENDING"
    product.footprint = [
        (lon0, lat0),
        (lon0 + 1, lat0),
        (lon0 + 1, lat0 + 1),
        (lon0, lat0 + 1),
        (lon0, lat0),
    ]
    product.s3_path = "/eodata/x"
    bbox_holder = type(
        "B", (), {"min_lon": lon0, "min_lat": lat0, "max_lon": lon0 + 1, "max_lat": lat0 + 1}
    )
    product.bbox = bbox_holder

    with session_scope(factory) as session:
        SceneRepository(session).upsert(product)


def test_scenes_endpoint_returns_footprints(client):
    factory = api._state["sessions"]
    _seed_scene(factory)

    body = client.get("/api/scenes").json()

    assert body["count"] == 1
    scene = body["scenes"][0]
    assert scene["satellite"] == "S1D"
    assert len(scene["footprint"]) == 5  # closed ring, drawable
    assert scene["polarisation"] == "DV"


def test_scenes_carry_their_age_for_the_coverage_overlay(client):
    _seed_scene(api._state["sessions"], hours_ago=5)

    scene = client.get("/api/scenes").json()["scenes"][0]

    # Where this is large, the platform is blind and must say so.
    assert 4.5 < scene["age_hours"] < 5.5


def test_scenes_report_the_last_radar_look(client):
    _seed_scene(api._state["sessions"])

    body = client.get("/api/scenes").json()

    assert body["last_look"] is not None
    assert body["attribution"].startswith("Contains modified Copernicus")


def test_scenes_outside_the_bbox_are_excluded(client):
    _seed_scene(api._state["sessions"], lon0=56.0, lat0=26.0)

    body = client.get("/api/scenes", params={"bbox": "0,50,5,55"}).json()

    assert body["count"] == 0
    assert body["last_look"] is None


def test_health_reports_known_scene_count(client):
    _seed_scene(api._state["sessions"])

    assert client.get("/api/health").json()["sar_scenes"] == 1


# --------------------------------------------------------------------------
# SAR detections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    """Shaped like core.sar.detect.Candidate."""

    lon: float
    lat: float
    length_m: float = 212.0
    width_m: float = 63.0
    heading_deg: float = 127.0
    pixel_count: int = 180
    peak_amplitude: float = 2911.0


def _seed_detection(factory, lon=56.5, lat=26.5, hours_ago=3):
    from core.db.repository import DetectionRepository, SceneRepository

    product_id = f"det-{hours_ago}"
    _seed_scene(factory, product_id=product_id, hours_ago=hours_ago)
    with session_scope(factory) as session:
        scene_id = SceneRepository(session).id_for(product_id)
        DetectionRepository(session).replace_in_bbox(
            scene_id,
            BBox(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1),
            [_Target(lon, lat)],
            polarisation="VH",
            detector="cfar",
            detector_version="test",
            confidence=1.0,
        )


def test_detections_endpoint_returns_radar_contacts(client):
    _seed_detection(api._state["sessions"])

    body = client.get("/api/detections").json()

    assert body["count"] == 1
    detection = body["detections"][0]
    assert (detection["lon"], detection["lat"]) == (56.5, 26.5)
    assert detection["length_m"] == 212.0
    assert detection["satellite"] == "S1D"
    assert detection["polarisation"] == "VH"
    assert detection["detector"] == "cfar"
    assert body["attribution"].startswith("Contains modified Copernicus")


def test_every_detection_says_whether_the_detector_was_measured_where_it_lies(client):
    """The model was trained and tested off the UAE only: elsewhere its accuracy is unknown."""
    _seed_detection(api._state["sessions"])  # in the Hormuz lanes study box
    _seed_detection(api._state["sessions"], lon=56.25, lat=27.03)  # Bandar Abbas anchorage

    found = client.get("/api/detections").json()["detections"]

    assert {d["lat"]: d["study_area"] for d in found} == {26.5: "hormuz_lanes", 27.03: None}
    assert {d["lat"]: d["measured_area"] for d in found} == {
        26.5: "hormuz_lanes",
        27.03: "bandar_abbas",
    }
    # Seeded as CFAR version "test": not a configuration measured against GFW.
    assert {d["tested"] for d in found} == {False}


def test_every_detection_carries_the_age_of_its_radar_pass(client):
    """A detection is a snapshot from the last pass; it must never look live."""
    _seed_detection(api._state["sessions"], hours_ago=5)

    detection = client.get("/api/detections").json()["detections"][0]

    assert 4.5 < detection["age_hours"] < 5.5
    assert detection["acq_time"].endswith("+00:00")


def test_detections_outside_the_bbox_are_excluded(client):
    _seed_detection(api._state["sessions"])

    body = client.get("/api/detections", params={"bbox": "0,50,5,55"}).json()

    assert body["count"] == 0


def test_detections_from_passes_older_than_the_window_are_excluded(client):
    _seed_detection(api._state["sessions"], hours_ago=24 * 20)

    assert client.get("/api/detections", params={"days": 14}).json()["count"] == 0


def test_detection_days_is_bounded(client):
    assert client.get("/api/detections", params={"days": 0}).status_code == 422


def test_health_reports_detection_count(client):
    _seed_detection(api._state["sessions"])

    assert client.get("/api/health").json()["detections"] == 1


@dataclass(frozen=True)
class _Point:
    """Shaped like a point-detector result: a position and a score, no hull."""

    lon: float
    lat: float
    confidence: float = 0.9
    length_m: None = None
    width_m: None = None
    heading_deg: None = None
    pixel_count: None = None
    peak_amplitude: None = None


def _seed_both_detectors(factory, lon=56.5, lat=26.5, hours_ago=3):
    """One pass that both CFAR and the model have searched."""
    from core.db.repository import DetectionRepository, SceneRepository

    _seed_detection(factory, lon=lon, lat=lat, hours_ago=hours_ago)
    with session_scope(factory) as session:
        scene_id = SceneRepository(session).id_for(f"det-{hours_ago}")
        DetectionRepository(session).replace_in_bbox(
            scene_id,
            BBox(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1),
            [_Point(lon, lat)],
            polarisation="VV+VH",
            detector="pointdet",
            detector_version="kaggle-v2",
            confidence=1.0,
        )


def test_detections_default_to_one_detector_per_pass(client):
    """A vessel seen by both detectors must be drawn and counted once."""
    _seed_both_detectors(api._state["sessions"])

    body = client.get("/api/detections").json()

    assert body["detector"] == "auto"
    assert body["count"] == 1
    assert body["detections"][0]["detector"] == "pointdet"


def test_detections_can_return_every_detector(client):
    _seed_both_detectors(api._state["sessions"])

    body = client.get("/api/detections", params={"detector": "all"}).json()

    assert body["detector"] == "all"
    assert sorted(d["detector"] for d in body["detections"]) == ["cfar", "pointdet"]


def test_detections_can_return_one_named_detector(client):
    _seed_both_detectors(api._state["sessions"])

    body = client.get("/api/detections", params={"detector": "cfar"}).json()

    assert body["detector"] == "cfar"
    assert [d["detector"] for d in body["detections"]] == ["cfar"]


def test_unknown_detector_is_rejected(client):
    assert client.get("/api/detections", params={"detector": "bogus"}).status_code == 422


def test_vessel_card_receives_reported_identity_and_type(client):
    with session_scope(api._state["sessions"]) as session:
        VesselRepository(session).upsert_static(
            VesselStatic(
                mmsi=MMSI,
                name="TEST TANKER",
                imo=1234567,
                callsign="TEST123",
                ship_type=80,
            )
        )
    vessel = client.get("/api/vessels").json()["vessels"][0]
    assert vessel["name"] == "TEST TANKER"
    assert vessel["imo"] == 1234567
    assert vessel["callsign"] == "TEST123"
    assert vessel["ship_type"] == 80


def test_absent_static_identity_remains_unknown(client):
    vessel = client.get("/api/vessels", params={"bbox": "0,45,20,55"}).json()["vessels"][0]
    assert vessel["name"] is None
    assert vessel["ship_type"] is None
    assert vessel["imo"] is None
    assert vessel["callsign"] is None
