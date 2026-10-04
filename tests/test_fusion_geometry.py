"""Box buffering for the GFW presence query around a pass.

Ships under way move up to about 13 km between hourly cells (P10, Fujairah,
2026-09-25), so presence is queried over the pass box buffered by about an
hour of travel; otherwise their H-1 and H+1 cells fall outside the query and
their detections look dark (PLAN §17, 2026-10-04 "Phase 6 feasibility").
"""

from __future__ import annotations

import math

import pytest

from core.db.spatial import BBox
from core.fusion.geometry import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR, buffer_box

BOX = BBox(56.43, 25.13, 56.63, 25.31)


def test_a_buffered_box_grows_by_the_distance_on_every_side():
    grown = buffer_box(BOX, 25_000.0)
    mid_lat = (BOX.min_lat + BOX.max_lat) / 2
    d_lat = 25_000.0 / M_PER_DEG_LAT
    d_lon = 25_000.0 / (M_PER_DEG_LON_EQUATOR * math.cos(math.radians(mid_lat)))
    assert grown.min_lat == pytest.approx(BOX.min_lat - d_lat)
    assert grown.max_lat == pytest.approx(BOX.max_lat + d_lat)
    assert grown.min_lon == pytest.approx(BOX.min_lon - d_lon)
    assert grown.max_lon == pytest.approx(BOX.max_lon + d_lon)


def test_a_zero_buffer_returns_the_same_box():
    assert buffer_box(BOX, 0.0) == BOX


def test_a_negative_buffer_is_rejected():
    with pytest.raises(ValueError):
        buffer_box(BOX, -1.0)
