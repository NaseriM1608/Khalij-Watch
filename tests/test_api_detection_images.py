"""Tests for the product model's detections with their satellite images.

The thesis console shows the vessels the deep-learning model found in
Sentinel-1 images, each with its image beside it: ``detector=product`` keeps
only the product model's rows (not CFAR, not the superseded run 1), every
detection says whether its image is on disk, and the image is served from
disk - a browser never causes a CDSE read.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from core.areas import box_around
from core.config import Settings
from core.db.repository import AnalysedWindowRepository, DetectionRepository, SceneRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from core.fusion.chips import chip_relpath
from services.api import main as api

NOW = dt.datetime.now(dt.UTC)
SCENE = "S1C_IW_GRDH_1SDV_20261003T021411_TEST_COG.SAFE"
BOX = box_around(56.53, 25.22, 20.0)
PNG = b"\x89PNG\r\n\x1a\nvessel"
WITH_IMAGE, WITHOUT_IMAGE, RUN1, CFAR = (
    (56.50, 25.20),
    (56.55, 25.25),
    (56.52, 25.22),
    (56.47, 25.17),
)


@dataclass(frozen=True)
class Extent:
    min_lon: float = 55.8
    min_lat: float = 24.0
    max_lon: float = 58.7
    max_lat: float = 27.0


@dataclass(frozen=True)
class Product:
    product_id: str = SCENE
    name: str = SCENE
    acq_start: dt.datetime = NOW - dt.timedelta(days=2)
    satellite: str = "S1C"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "DESCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
    footprint: tuple = ((55.8, 24.0), (58.7, 24.0), (58.7, 27.0), (55.8, 27.0), (55.8, 24.0))
    bbox: Extent = Extent()


@dataclass(frozen=True)
class Point:
    lon: float
    lat: float
    confidence: float = 0.83
    length_m: float | None = None
    width_m: float | None = None
    heading_deg: float | None = None
    pixel_count: int | None = None
    peak_amplitude: float | None = None


def store(session, scene_id, box, points, detector, version, pol):
    DetectionRepository(session).replace_in_bbox(
        scene_id,
        box,
        [Point(*p) for p in points],
        polarisation=pol,
        detector=detector,
        detector_version=version,
        confidence=1.0,
    )
    AnalysedWindowRepository(session).record(
        scene_id,
        box,
        detector=detector,
        detector_version=version,
        polarisation=pol,
        detections=len(points),
    )


@pytest.fixture
def client(tmp_path):
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        SceneRepository(session).upsert(Product())
        scene_id = SceneRepository(session).id_for(SCENE)
        store(session, scene_id, BOX, [WITH_IMAGE, WITHOUT_IMAGE], "pointdet", "kaggle-v2", "DV")
        store(
            session,
            scene_id,
            BBox(56.51, 25.21, 56.53, 25.23),
            [RUN1],
            "pointdet",
            "kaggle-v1",
            "DV",
        )
        store(session, scene_id, BOX, [CFAR], "cfar", "L4.4-g24-t40-p1e-06-h0.25", "VH")
    image = tmp_path / "chips" / chip_relpath(SCENE, *WITH_IMAGE, "vv")
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG)
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


def product_detections(client) -> dict:
    found = client.get("/api/detections", params={"detector": "product"}).json()["detections"]
    return {(d["lon"], d["lat"]): d for d in found}


def test_only_the_product_models_detections_are_returned(client):
    found = product_detections(client)
    assert set(found) == {WITH_IMAGE, WITHOUT_IMAGE}
    assert {(d["detector"], d["detector_version"]) for d in found.values()} == {
        ("pointdet", "kaggle-v2")
    }


def test_every_detection_says_whether_its_image_is_on_disk(client):
    found = product_detections(client)
    assert found[WITH_IMAGE]["chip"]["pols"] == ["vv"]
    assert found[WITH_IMAGE]["chip"]["north_up"] is True
    assert found[WITHOUT_IMAGE]["chip"] is None


def test_a_detections_image_is_served_from_disk(client):
    found = product_detections(client)
    response = client.get(f"/api/detections/{found[WITH_IMAGE]['id']}/chip/vv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-cache"
    assert response.content == PNG


def test_an_image_not_on_disk_or_an_unknown_detection_is_not_found(client):
    found = product_detections(client)
    assert client.get(f"/api/detections/{found[WITH_IMAGE]['id']}/chip/vh").status_code == 404
    assert client.get(f"/api/detections/{found[WITHOUT_IMAGE]['id']}/chip/vv").status_code == 404
    assert client.get("/api/detections/999999/chip/vv").status_code == 404
    assert client.get(f"/api/detections/{2**64}/chip/vv").status_code == 422
    assert client.get(f"/api/detections/{found[WITH_IMAGE]['id']}/chip/hh").status_code == 422


def test_only_the_product_models_searches_are_returned(client):
    body = client.get("/api/analysed", params={"detector": "product"}).json()
    assert {(w["detector"], w["detector_version"]) for w in body["windows"]} == {
        ("pointdet", "kaggle-v2")
    }
    assert client.get("/api/analysed").json()["count"] == 3


def test_detections_of_fused_passes_are_shown_whatever_their_age(client):
    """Their dark-vessel candidates and searched areas are always shown; so are their detections."""
    from core.db.repository import EventRepository, NewEvent

    with session_scope(api._state["sessions"]) as session:
        SceneRepository(session).upsert(
            Product(
                product_id="S1D_FUSED_COG.SAFE",
                name="S1D_FUSED_COG.SAFE",
                acq_start=NOW - dt.timedelta(days=25),
            )
        )
        fused = SceneRepository(session).id_for("S1D_FUSED_COG.SAFE")
        store(session, fused, BOX, [(56.58, 25.28)], "pointdet", "kaggle-v2", "DV")
        SceneRepository(session).upsert(
            Product(
                product_id="S1D_OLD_COG.SAFE",
                name="S1D_OLD_COG.SAFE",
                acq_start=NOW - dt.timedelta(days=25),
            )
        )
        old = SceneRepository(session).id_for("S1D_OLD_COG.SAFE")
        store(session, old, BOX, [(56.59, 25.29)], "pointdet", "kaggle-v2", "DV")
        EventRepository(session).replace_for_scene(
            fused,
            "fusion-v1",
            [
                NewEvent(
                    event_type="DARK_VESSEL",
                    ts=NOW - dt.timedelta(days=25),
                    lon=56.58,
                    lat=25.28,
                    severity="low",
                    detection_id=None,
                    evidence={"status": "dark"},
                )
            ],
        )
    found = product_detections(client)
    assert (56.58, 25.28) in found
    assert (56.59, 25.29) not in found


def test_an_image_asked_for_with_another_detections_key_is_not_served(client):
    """A re-run can give a detection's id to another one: a card holding the old key gets nothing."""
    found = product_detections(client)
    detection = found[WITH_IMAGE]
    good = f"{SCENE}/{WITH_IMAGE[1]}_{WITH_IMAGE[0]}/x"
    stale = f"{SCENE}/25.0_56.0/x"
    url = f"/api/detections/{detection['id']}/chip/vv"
    assert client.get(url, params={"v": good}).status_code == 200
    assert client.get(url, params={"v": stale}).status_code == 404
