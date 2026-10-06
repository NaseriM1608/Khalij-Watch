"""Tests for the timeline scrubber's API: the console as of a past moment, and every pass per box.

PLAN §10 rule 4: drag back through time; each area's passes appear as bands
showing when it was actually observed. ``as_of`` replaces "now": only passes
acquired by then count, and ages are measured from it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from core.areas import IRAN_AREAS, STUDY_AREAS, box_around
from core.db.repository import (
    AnalysedWindowRepository,
    DetectionRepository,
    EventRepository,
    NewEvent,
    SceneRepository,
)
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from services.api import main as api

NOW = dt.datetime.now(dt.UTC).replace(microsecond=0)
AGES = {"S1A_OLD": 20, "S1C_MID": 10, "S1D_NEW": 2}  # days before now
FUJAIRAH = box_around(*STUDY_AREAS["fujairah"])
WIDE = ((50.0, 24.0), (58.7, 24.0), (58.7, 30.0), (50.0, 30.0), (50.0, 24.0))
# A pass imaging only the inner Gulf: not over Fujairah or Bandar Abbas.
WEST = ((50.0, 26.0), (54.0, 26.0), (54.0, 30.0), (50.0, 30.0), (50.0, 26.0))


@dataclass(frozen=True)
class Extent:
    min_lon: float = 50.0
    min_lat: float = 24.0
    max_lon: float = 58.7
    max_lat: float = 30.0


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    acq_start: dt.datetime
    footprint: tuple = WIDE
    satellite: str = "S1C"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "DESCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
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


def acquired(name: str) -> dt.datetime:
    return NOW - dt.timedelta(days=AGES.get(name, 5))


@pytest.fixture
def client():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        scenes = SceneRepository(session)
        for name in AGES:
            scenes.upsert(Product(name, name, acquired(name)))
        scenes.upsert(Product("S1D_WEST", "S1D_WEST", acquired("S1D_WEST"), footprint=WEST))
        ids = {name: scenes.id_for(name) for name in [*AGES, "S1D_WEST"]}
        for name in ("S1C_MID", "S1D_NEW"):
            DetectionRepository(session).replace_in_bbox(
                ids[name],
                FUJAIRAH,
                [Peak(56.53, 25.22)],
                polarisation="DV",
                detector="pointdet",
                detector_version="kaggle-v2",
                confidence=1.0,
                shore_m=[5000.0],
            )
            AnalysedWindowRepository(session).record(
                ids[name],
                FUJAIRAH,
                detector="pointdet",
                detector_version="kaggle-v2",
                polarisation="DV",
                detections=1,
            )
        EventRepository(session).replace_for_scene(
            ids["S1C_MID"],
            "fusion-v1",
            [
                NewEvent(
                    event_type="DARK_VESSEL",
                    ts=acquired("S1C_MID"),
                    lon=56.53,
                    lat=25.22,
                    severity="low",
                    detection_id=None,
                    evidence={"status": "dark"},
                )
            ],
        )
    with TestClient(api.app) as test_client:
        api._state["sessions"] = factory
        yield test_client
    api._state.clear()


AS_OF = (NOW - dt.timedelta(days=5)).isoformat()


def test_detections_as_of_a_past_moment_hold_only_passes_acquired_by_then(client):
    body = client.get(
        "/api/detections", params={"detector": "product", "as_of": AS_OF, "days": 14}
    ).json()
    assert [d["product_name"] for d in body["detections"]] == ["S1C_MID"]
    assert body["detections"][0]["age_hours"] == pytest.approx(5 * 24, abs=0.1)
    assert body["as_of"] == AS_OF


def test_scenes_searches_and_events_as_of_a_past_moment(client):
    scenes = client.get("/api/scenes", params={"as_of": AS_OF, "days": 14}).json()["scenes"]
    assert {s["product_name"] for s in scenes} == {"S1C_MID", "S1D_WEST"}
    windows = client.get(
        "/api/analysed", params={"as_of": AS_OF, "days": 14, "detector": "product"}
    ).json()["windows"]
    assert [w["product_name"] for w in windows] == ["S1C_MID"]
    events = client.get("/api/events", params={"type": "DARK_VESSEL", "as_of": AS_OF}).json()
    assert [e["age_hours"] for e in events["events"]] == [pytest.approx(5 * 24, abs=0.1)]
    earlier = (NOW - dt.timedelta(days=15)).isoformat()
    assert client.get("/api/events", params={"as_of": earlier}).json()["events"] == []


def test_a_moment_in_the_future_means_now(client):
    later = (NOW + dt.timedelta(days=3)).isoformat()
    body = client.get("/api/detections", params={"detector": "product", "as_of": later}).json()
    assert {d["product_name"] for d in body["detections"]} == {"S1C_MID", "S1D_NEW"}


def test_the_timeline_lists_every_pass_over_each_measured_box_oldest_first(client):
    body = client.get("/api/timeline", params={"days": 30}).json()

    areas = {a["name"]: a for a in body["areas"]}
    assert list(areas) == [*STUDY_AREAS, *IRAN_AREAS]
    assert areas["bandar_abbas"]["region"] == "iran"
    fujairah = areas["fujairah"]
    assert [p["product_name"] for p in fujairah["passes"]] == ["S1A_OLD", "S1C_MID", "S1D_NEW"]
    flags = {
        p["product_name"]: (p["searched"], p["detections"], p["fused"]) for p in fujairah["passes"]
    }
    assert flags == {
        "S1A_OLD": (False, 0, False),
        "S1C_MID": (True, 1, True),
        "S1D_NEW": (True, 1, False),
    }
    assert [p["product_name"] for p in areas["bandar_abbas"]["passes"]] == [
        "S1A_OLD",
        "S1C_MID",
        "S1D_NEW",
    ]
    assert "S1D_WEST" in [p["product_name"] for p in areas["kharg"]["passes"]]
    assert "S1D_WEST" not in [p["product_name"] for p in fujairah["passes"]]


def test_the_timeline_window_is_counted_back_from_now(client):
    body = client.get("/api/timeline", params={"days": 15}).json()
    fujairah = next(a for a in body["areas"] if a["name"] == "fujairah")
    assert [p["product_name"] for p in fujairah["passes"]] == ["S1C_MID", "S1D_NEW"]
