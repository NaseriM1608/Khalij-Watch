"""Tests for the study boxes: where the detector's accuracy was measured.

The product model was trained, validated and tested on five boxes off the
UAE (2024-25, Sentinel-1A) and checked there on Sentinel-1C/1D (2026-09).
Detections elsewhere - the Iranian coast - come from an unmeasured detector,
and the console must be able to say so.
"""

from __future__ import annotations

import pytest

from core.areas import (
    IRAN_AREAS,
    STUDY_AREAS,
    box_around,
    clip_to_box,
    inside_study_box,
    is_tested,
    ring_area,
    study_area_of,
)
from core.geo import haversine_m


def test_the_five_study_boxes_are_the_ones_the_model_was_measured_on():
    assert set(STUDY_AREAS) == {
        "fujairah",
        "khor_fakkan",
        "sharjah_dubai",
        "ras_al_khaimah",
        "hormuz_lanes",
    }
    assert STUDY_AREAS["hormuz_lanes"] == (56.55, 26.50, 40.0)


def test_a_box_is_the_requested_size_on_the_ground():
    box = box_around(56.53, 25.22, 20.0)
    assert box.min_lon < 56.53 < box.max_lon and box.min_lat < 25.22 < box.max_lat
    width = haversine_m(box.min_lon, 25.22, box.max_lon, 25.22)
    height = haversine_m(56.53, box.min_lat, 56.53, box.max_lat)
    assert width == pytest.approx(20_000, rel=0.01)
    assert height == pytest.approx(20_000, rel=0.01)


@pytest.mark.parametrize(
    ("lon", "lat", "area"),
    [
        (56.53, 25.22, "fujairah"),
        (56.60, 25.30, "fujairah"),
        (56.55, 26.62, "hormuz_lanes"),
        (56.25, 27.03, None),  # Bandar Abbas anchorage
        (50.30, 29.25, None),  # Kharg Island
    ],
)
def test_a_point_is_placed_in_its_study_box_or_none(lon, lat, area):
    assert study_area_of(lon, lat) == area


def test_the_iranian_boxes_are_the_consoles_seven_and_none_overlaps_a_study_box():
    """Held-out boxes for validating the model on the Iranian coast, never study boxes."""
    assert IRAN_AREAS == {
        "bandar_abbas": (56.25, 27.03, 20.0),
        "larak_hormuz": (56.45, 26.85, 20.0),
        "kish": (53.98, 26.53, 20.0),
        "assaluyeh": (52.55, 27.40, 20.0),
        "bushehr": (50.72, 28.95, 20.0),
        "kharg": (50.30, 29.25, 20.0),
        "jask": (57.80, 25.60, 20.0),
    }
    for lon, lat, size in IRAN_AREAS.values():
        box = box_around(lon, lat, size)
        assert study_area_of(lon, lat) is None
        for study in STUDY_AREAS.values():
            other = box_around(*study)
            assert (
                box.max_lon < other.min_lon
                or other.max_lon < box.min_lon
                or box.max_lat < other.min_lat
                or other.max_lat < box.min_lat
            )


# --------------------------------------------------------------------------
# The searched area: the requested box clipped to the pass footprint
# --------------------------------------------------------------------------

BOX = box_around(56.53, 25.22, 20.0)


def ring_of(min_lon, min_lat, max_lon, max_lat):
    return [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ]


def box_ring(box):
    return ring_of(box.min_lon, box.min_lat, box.max_lon, box.max_lat)


def test_a_footprint_covering_the_box_leaves_the_whole_box():
    clipped = clip_to_box(ring_of(55.0, 24.0, 58.0, 27.0), BOX)
    assert ring_area(clipped) == pytest.approx(ring_area(box_ring(BOX)), rel=1e-9)


def test_a_swath_edge_through_the_box_leaves_only_the_imaged_part():
    """The Fujairah study box on the 02:14 UTC passes: the swath ends mid-box."""
    middle = (BOX.min_lon + BOX.max_lon) / 2
    clipped = clip_to_box(ring_of(55.0, 24.0, middle, 27.0), BOX)
    box_area = ring_area(box_ring(BOX))
    assert ring_area(clipped) == pytest.approx(box_area / 2, rel=1e-6)
    assert max(lon for lon, _ in clipped) == pytest.approx(middle)


def test_a_slanted_swath_edge_is_clipped_too():
    # A swath edge running diagonally across the box's north-east corner.
    mid_lon = (BOX.min_lon + BOX.max_lon) / 2
    mid_lat = (BOX.min_lat + BOX.max_lat) / 2
    # The edge runs through the midpoints of the box's east and north sides,
    # extended beyond them, so it cuts off one eighth of the box.
    d_lon, d_lat = mid_lon - BOX.max_lon, BOX.max_lat - mid_lat
    swath = [
        (BOX.min_lon - 0.1, BOX.min_lat - 0.1),
        (BOX.max_lon + 0.1, BOX.min_lat - 0.1),
        (BOX.max_lon - d_lon, mid_lat - d_lat),
        (mid_lon + d_lon, BOX.max_lat + d_lat),
        (BOX.min_lon - 0.1, BOX.max_lat + 0.1),
    ]
    clipped = clip_to_box(swath, BOX)
    assert ring_area(clipped) == pytest.approx(7 / 8 * ring_area(box_ring(BOX)), rel=1e-9)
    tolerance = 1e-12
    assert all(
        BOX.min_lon - tolerance <= lon <= BOX.max_lon + tolerance
        and BOX.min_lat - tolerance <= lat <= BOX.max_lat + tolerance
        for lon, lat in clipped
    )


def test_a_footprint_missing_the_box_leaves_nothing():
    assert clip_to_box(ring_of(50.0, 28.0, 51.0, 29.0), BOX) == []


def test_a_searched_area_counts_as_a_study_area_only_if_wholly_inside_one():
    assert inside_study_box(clip_to_box(ring_of(55.0, 24.0, 58.0, 27.0), BOX)) == "fujairah"
    straddling = box_around(56.53, 25.22, 40.0)  # reaches into Khor Fakkan and beyond
    assert inside_study_box(box_ring(straddling)) is None
    assert inside_study_box(box_ring(box_around(56.25, 27.03, 20.0))) is None
    assert inside_study_box([]) is None


def test_only_the_detector_configurations_measured_against_gfw_count_as_tested():
    assert is_tested("pointdet", "kaggle-v2", "DV")  # run 2, the product model
    assert is_tested("cfar", "L4.4-g24-t40-p1e-06-h0.25", "VH")  # the CFAR baseline
    assert not is_tested("pointdet", "kaggle-v1", "DV")  # run 1: superseded, lost to CFAR
    assert not is_tested("cfar", "L4.4-g24-t40-p1e-06-h0.25", "VV")  # never compared on VV
    assert not is_tested("cfar", "L4.4-g24-t40-p0.001-h0.25", "VH")
