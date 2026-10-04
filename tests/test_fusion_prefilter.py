"""Pre-filters on our SAR detections before the dark-rate count.

A detection only counts towards the dark rate if it is a vessel candidate in
open, imaged water: duplicates of the same ship (seam peaks, CFAR fragments
about 170 m apart) are merged, and detections near the shore, on fixed
infrastructure or at the edge of the processed box are set aside with a
reason (PLAN §17, 2026-10-04 "Phase 6 feasibility", "An honest dark rate").
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from core.db.spatial import BBox
from core.fusion.prefilter import (
    DUPLICATE,
    EDGE,
    INFRASTRUCTURE,
    SHORE,
    SarDetection,
    distance_to_box_edge_m,
    distance_to_land_m,
    merge_duplicates,
    prefilter,
)
from core.sar.coastline import LandPolygon
from core.sar.infrastructure import InfrastructureMask, Platform

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0
LAT = 25.2


def east_m(deg: float, lat: float = LAT) -> float:
    return deg * M_PER_DEG_LON_EQ * math.cos(math.radians(lat))


def det(lon: float, lat: float = LAT, score: float = 0.9) -> SarDetection:
    return SarDetection(lon=lon, lat=lat, score=score)


ISLAND = LandPolygon(np.array([[56.30, 25.10], [56.40, 25.10], [56.40, 25.30], [56.30, 25.30]]))


# ---------------------------------------------------------------------------
# Duplicate merging
# ---------------------------------------------------------------------------


def test_a_weaker_detection_within_100_m_merges_into_the_stronger():
    offset = 50 / east_m(1.0)
    assert merge_duplicates([det(56.5, score=0.7), det(56.5 + offset, score=0.9)]) == [1, 1]


def test_detections_150_m_apart_both_stay():
    offset = 150 / east_m(1.0)
    assert merge_duplicates([det(56.5), det(56.5 + offset)]) == [0, 1]


def test_merging_does_not_chain_beyond_the_radius():
    step = 80 / east_m(1.0)
    chain = [det(56.5, score=0.9), det(56.5 + step, score=0.8), det(56.5 + 2 * step, score=0.7)]
    assert merge_duplicates(chain) == [0, 0, 2]


# ---------------------------------------------------------------------------
# Distance to land and to the box edge
# ---------------------------------------------------------------------------


def test_a_point_on_land_is_at_distance_zero():
    assert distance_to_land_m(56.35, 25.20, [ISLAND]) == 0.0


def test_distance_to_land_is_to_the_nearest_coast_segment():
    assert distance_to_land_m(56.41, 25.20, [ISLAND]) == pytest.approx(
        east_m(0.01, 25.20), rel=1e-3
    )


def test_a_point_in_a_lagoon_is_measured_to_the_lagoon_shore():
    lagoon = np.array([[56.34, 25.18], [56.36, 25.18], [56.36, 25.22], [56.34, 25.22]])
    atoll = LandPolygon(ISLAND.exterior, holes=(lagoon,))
    assert distance_to_land_m(56.35, 25.20, [atoll]) == pytest.approx(east_m(0.01, 25.20), rel=1e-3)


def test_no_land_means_infinite_distance():
    assert distance_to_land_m(56.5, 25.2, []) == math.inf


def test_distance_to_the_box_edge_is_to_the_nearest_side():
    box = BBox(56.40, 25.10, 56.60, 25.30)
    assert distance_to_box_edge_m(56.401, 25.20, box) == pytest.approx(
        east_m(0.001, 25.20), rel=1e-3
    )
    assert distance_to_box_edge_m(56.50, 25.299, box) == pytest.approx(
        0.001 * M_PER_DEG_LAT, rel=1e-3
    )


def test_a_point_outside_the_box_is_at_distance_zero():
    assert distance_to_box_edge_m(56.70, 25.20, BBox(56.40, 25.10, 56.60, 25.30)) == 0.0


# ---------------------------------------------------------------------------
# The combined pre-filter
# ---------------------------------------------------------------------------


def test_each_detection_gets_its_first_exclusion_reason_or_none():
    box = BBox(56.40, 25.10, 56.70, 25.30)
    platform = Platform(lon=56.60, lat=25.15, installed=None, last_observed=None)
    detections = [
        det(56.55, score=0.95),  # clean
        det(56.55 + 50 / east_m(1.0), score=0.80),  # duplicate of 0
        det(56.405, score=0.9),  # 0.005 deg off the island: within 1 km of shore (and the box edge)
        det(56.601, 25.15, score=0.9),  # on the platform
        det(56.699, 25.25, score=0.9),  # 100 m inside the east edge
        det(56.65, 25.25, score=0.9),  # clean
    ]
    results = prefilter(
        detections,
        land=[ISLAND],
        infrastructure=InfrastructureMask([platform], radius_m=300.0),
        box=box,
        shore_m=1000.0,
        edge_m=500.0,
    )
    assert [r.reason for r in results] == [None, DUPLICATE, SHORE, INFRASTRUCTURE, EDGE, None]
    assert results[1].merged_into == 0
    assert [r.kept for r in results] == [True, False, False, False, False, True]


def test_the_pre_filter_keeps_measured_distances_for_evidence():
    (result,) = prefilter(
        [det(56.55)],
        land=[ISLAND],
        infrastructure=InfrastructureMask([], radius_m=300.0),
        box=BBox(56.40, 25.10, 56.70, 25.30),
        shore_m=1000.0,
        edge_m=500.0,
    )
    assert result.shore_m == pytest.approx(east_m(0.15, LAT), rel=1e-3)
    assert result.edge_m == pytest.approx(0.10 * M_PER_DEG_LAT, rel=1e-3)


def test_infrastructure_is_applied_for_the_pass_date_by_the_caller():
    old = Platform(lon=56.60, lat=25.15, installed=None, last_observed=dt.date(2019, 1, 1))
    mask = InfrastructureMask([old], radius_m=300.0).for_date(dt.date(2026, 9, 25))
    (result,) = prefilter(
        [det(56.601, 25.15)],
        land=[],
        infrastructure=mask,
        box=BBox(56.40, 25.10, 56.70, 25.30),
        shore_m=1000.0,
        edge_m=500.0,
    )
    assert result.kept
