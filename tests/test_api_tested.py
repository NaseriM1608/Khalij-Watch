"""Tests for what /api/detections claims about each detection's accuracy.

The product model was measured against GFW in the five UAE study boxes and
the seven Iranian boxes (PLAN §17, 2026-10-05), away from the coast: GFW
reports no vessel within about 1 km of land. A detection is "tested" only in
a measured box, at least 1 km from land, and then carries its region's
measured precision and recall.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from core.areas import box_around
from core.db.repository import DetectionRepository, SceneRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from services.api import main as api

NOW = dt.datetime.now(dt.UTC)
SCENE = "S1C_IW_GRDH_1SDV_20261003T021411_TEST_COG.SAFE"


@dataclass(frozen=True)
class Extent:
    min_lon: float = 50.0
    min_lat: float = 24.0
    max_lon: float = 58.7
    max_lat: float = 30.0


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
    footprint: tuple = ((50.0, 24.0), (58.7, 24.0), (58.7, 30.0), (50.0, 30.0), (50.0, 24.0))
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


# (lon, lat): distance to land in metres, None if unknown.
FAR_IRAN, NEAR_IRAN, UNKNOWN_IRAN = (56.25, 27.03), (56.26, 27.04), (56.27, 27.05)
FAR_UAE = (56.53, 25.22)
OPEN_GULF = (52.00, 26.00)
SHORE = {FAR_IRAN: 1500.0, NEAR_IRAN: 400.0, UNKNOWN_IRAN: None, FAR_UAE: 5000.0, OPEN_GULF: 5000.0}


@pytest.fixture
def client():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        SceneRepository(session).upsert(Product())
        scene_id = SceneRepository(session).id_for(SCENE)
        for centre, points in (
            ((56.25, 27.03), [FAR_IRAN, NEAR_IRAN, UNKNOWN_IRAN]),
            ((56.53, 25.22), [FAR_UAE]),
            ((52.00, 26.00), [OPEN_GULF]),
        ):
            DetectionRepository(session).replace_in_bbox(
                scene_id,
                box_around(*centre, 20.0),
                [Peak(*p) for p in points],
                polarisation="DV",
                detector="pointdet",
                detector_version="kaggle-v2",
                confidence=1.0,
                shore_m=[SHORE[p] for p in points],
            )
    with TestClient(api.app) as test_client:
        api._state["sessions"] = factory
        yield test_client
    api._state.clear()


def detections(client) -> dict:
    found = client.get("/api/detections", params={"detector": "product"}).json()["detections"]
    return {(d["lon"], d["lat"]): d for d in found}


def test_a_detection_far_from_land_in_an_iranian_box_is_tested_with_irans_accuracy(client):
    found = detections(client)[FAR_IRAN]
    assert found["tested"] is True
    assert found["measured_area"] == "bandar_abbas"
    assert found["accuracy"] == {
        "area": "bandar_abbas",
        "region": "iran",
        "precision": 0.51,
        "recall": 0.91,
    }
    assert found["shore_m"] == 1500


def test_a_detection_in_a_study_box_gets_the_uae_accuracy(client):
    found = detections(client)[FAR_UAE]
    assert found["tested"] is True
    assert (found["accuracy"]["region"], found["accuracy"]["precision"]) == ("uae", 0.59)


@pytest.mark.parametrize("where", [NEAR_IRAN, UNKNOWN_IRAN, OPEN_GULF])
def test_near_shore_unknown_distance_or_outside_the_boxes_is_not_tested(client, where):
    found = detections(client)[where]
    assert found["tested"] is False
    assert found["accuracy"] is None
    assert found["shore_m"] == (None if SHORE[where] is None else round(SHORE[where]))
