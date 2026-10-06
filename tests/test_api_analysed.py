"""Tests for /api/analysed: where a detector searched, and whether it was tested there.

Outside every searched area no ring means "not searched", and inside one no
ring is still not proof of no ship. The searched area is the requested box
clipped to the pass footprint: part of a box can lie beyond the swath. A
search counts as tested only for a detector configuration measured against
GFW, wholly inside one of the five study boxes (PLAN §10, coverage honesty).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.areas import box_around
from core.config import Settings
from core.db.repository import (
    AnalysedWindowRepository,
    EventRepository,
    NewEvent,
    SceneRepository,
)
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from services.api import main as api

NOW = dt.datetime.now(dt.UTC)
FUJAIRAH = box_around(56.53, 25.22, 20.0)
MIDDLE = (FUJAIRAH.min_lon + FUJAIRAH.max_lon) / 2
WIDE = ((50.0, 24.0), (58.7, 24.0), (58.7, 30.0), (50.0, 30.0), (50.0, 24.0))
# A swath whose east edge runs through the middle of the Fujairah box.
WEST_HALF = ((50.0, 24.0), (MIDDLE, 24.0), (MIDDLE, 30.0), (50.0, 30.0), (50.0, 24.0))


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


def search(windows, scene_id, box, detector="pointdet", version="kaggle-v2", pol="DV"):
    windows.record(
        scene_id, box, detector=detector, detector_version=version, polarisation=pol, detections=0
    )


@pytest.fixture
def client():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        scenes = SceneRepository(session)
        for name, age_days, footprint in (
            ("S1C_NEW", 2, WIDE),
            ("S1D_RUN1", 4, WIDE),
            ("S1D_HALF", 3, WEST_HALF),
            ("S1D_OLD", 30, WIDE),
            ("S1D_FUSED", 25, WIDE),
        ):
            scenes.upsert(Product(name, name, NOW - dt.timedelta(days=age_days), footprint))
        new, run1, half, old, fused = (
            scenes.id_for(n) for n in ("S1C_NEW", "S1D_RUN1", "S1D_HALF", "S1D_OLD", "S1D_FUSED")
        )
        windows = AnalysedWindowRepository(session)
        search(windows, new, FUJAIRAH)
        search(windows, run1, FUJAIRAH, version="kaggle-v1")  # run 1: superseded
        search(windows, new, box_around(56.53, 25.22, 40.0))  # straddles the study box
        search(windows, new, box_around(56.25, 27.03, 20.0))  # Bandar Abbas
        search(
            windows, half, FUJAIRAH, detector="cfar", version="L4.4-g24-t40-p1e-06-h0.25", pol="VH"
        )
        search(windows, old, box_around(50.30, 29.25, 20.0))
        search(
            windows, fused, FUJAIRAH, detector="cfar", version="L4.4-g24-t40-p1e-06-h0.25", pol="VH"
        )
        EventRepository(session).replace_for_scene(
            fused,
            "fusion-v1",
            [
                NewEvent(
                    event_type="DARK_VESSEL",
                    ts=NOW - dt.timedelta(days=25),
                    lon=56.5,
                    lat=25.2,
                    severity="low",
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
        data_dir=Path("data"),
    )
    with TestClient(api.app) as test_client:
        api._state["sessions"] = factory
        api._state["settings"] = settings
        yield test_client
    api._state.clear()


def windows_of(client, **params) -> dict:
    body = client.get("/api/analysed", params=params).json()
    return {
        (
            w["product_name"],
            w["detector_version"],
            round(w["bbox"]["min_lon"], 2),  # boxes of one pass differ by position
        ): w
        for w in body["windows"]
    }


def test_a_search_is_tested_only_for_a_tested_detector_wholly_inside_a_measured_box(client):
    found = windows_of(client)
    west = round(FUJAIRAH.min_lon, 2)
    product = found[("S1C_NEW", "kaggle-v2", west)]
    assert (product["measured_area"], product["tested"]) == ("fujairah", True)
    assert found[("S1D_RUN1", "kaggle-v1", west)]["tested"] is False
    straddling = found[("S1C_NEW", "kaggle-v2", round(box_around(56.53, 25.22, 40.0).min_lon, 2))]
    assert (straddling["measured_area"], straddling["tested"]) == (None, False)
    bandar = [w for w in found.values() if w["bbox"]["min_lat"] < 27.03 < w["bbox"]["max_lat"]]
    # Measured against GFW since 2026-10-05 (PLAN §17, the Iranian coast).
    assert [(w["measured_area"], w["tested"]) for w in bandar] == [("bandar_abbas", True)]


def test_the_searched_area_stops_at_the_swath_edge(client):
    found = windows_of(client)
    west = round(FUJAIRAH.min_lon, 2)
    half = found[("S1D_HALF", "L4.4-g24-t40-p1e-06-h0.25", west)]
    assert half["searched_share"] == pytest.approx(0.5, abs=0.01)
    assert max(lon for lon, _ in half["searched"]) == pytest.approx(MIDDLE)
    whole = found[("S1C_NEW", "kaggle-v2", west)]
    assert whole["searched_share"] == pytest.approx(1.0, abs=1e-6)
    assert (half["polarisation"], half["tested"]) == ("VH", True)


def test_older_passes_drop_out_unless_their_candidates_are_shown(client):
    names = {k[0] for k in windows_of(client)}
    assert "S1D_OLD" not in names
    assert "S1D_FUSED" in names
    assert "S1D_OLD" in {k[0] for k in windows_of(client, days=60)}


def test_every_search_says_how_old_its_pass_is(client):
    body = client.get("/api/analysed").json()
    ages = {w["product_name"]: w["age_hours"] for w in body["windows"]}
    assert 47 < ages["S1C_NEW"] < 49
    assert "Copernicus" in body["attribution"]


def test_a_bad_bbox_is_rejected(client):
    assert client.get("/api/analysed", params={"bbox": "nonsense"}).status_code == 400
