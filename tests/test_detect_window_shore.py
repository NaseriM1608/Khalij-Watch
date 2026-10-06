"""Tests for the distance to land that every stored detection keeps.

GFW reports no vessel within about 1 km of land, so the detector's accuracy
there was never measured (PLAN §17, 2026-10-05 Iranian coast); the console
can say so only if each detection knows how far it lies from land.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pytest

from core.areas import box_around
from core.db.repository import DetectionRepository
from core.db.session import create_db_engine, make_session_factory, session_scope
from core.sar.coastline import LandPolygon
from scripts import detect_window

# Land west of 56.40E: a straight north-south coast.
LAND = [LandPolygon(np.array([[55.0, 24.0], [56.40, 24.0], [56.40, 27.0], [55.0, 27.0]]))]
BOX = box_around(56.53, 25.22, 20.0)


@dataclass(frozen=True)
class Extent:
    min_lon: float = 55.8
    min_lat: float = 24.0
    max_lon: float = 58.7
    max_lat: float = 27.5


@dataclass(frozen=True)
class Product:
    product_id: str = "S1C_A"
    name: str = "S1C_A"
    acq_start: dt.datetime = dt.datetime(2026, 10, 3, 2, 14, 40, tzinfo=dt.UTC)
    satellite: str = "S1C"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "DESCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
    footprint: tuple = ((55.8, 24.0), (58.7, 24.0), (58.7, 27.5), (55.8, 27.5), (55.8, 24.0))
    bbox: Extent = Extent()
    online: bool = True


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


def test_shore_distances_are_metres_to_the_nearest_land_up_to_the_reach():
    points = [Peak(56.41, 25.2), Peak(56.43, 25.2), Peak(56.9, 25.2), Peak(56.3, 25.2)]

    found = detect_window.shore_distances(points, LAND, reach_m=5000.0)

    assert found[0] == pytest.approx(1007, rel=0.02)
    assert found[1] == pytest.approx(3021, rel=0.02)
    assert found[2] == 5000.0  # at least the reach: land farther away was not loaded
    assert found[3] == 0.0  # on land


def test_without_land_every_distance_is_the_reach():
    assert detect_window.shore_distances([Peak(56.41, 25.2)], [], reach_m=5000.0) == [5000.0]


def test_stored_detections_keep_their_distance_to_land(tmp_path, monkeypatch):
    monkeypatch.setattr(detect_window, "land_around", lambda _box, _reach_m: LAND)
    url = f"sqlite+pysqlite:///{(tmp_path / 'khalij.db').as_posix()}"

    detect_window.store_detections(
        url, Product(), BOX, [Peak(56.44, 25.2)], "dv", "kaggle-v2", detector="pointdet"
    )

    with session_scope(make_session_factory(create_db_engine(url))) as session:
        (stored,) = DetectionRepository(session).in_bbox(BOX, detector="all")
    assert stored.shore_m == pytest.approx(4028, rel=0.02)
