"""Tests for persistent spots: where something is detected pass after pass.

GFW removes such objects as infrastructure; our platform mask (OPD) misses
jetties, moorings, buoys and unmapped platforms, and ships moored for months
look the same (PLAN §17, 2026-10-05/06). A spot is fixed by a rule set before
any result: detected within RADIUS_M on at least MIN_PASSES passes and on at
least MIN_SHARE of the box's archive passes.
"""

from __future__ import annotations

import math

import pytest

from core.persistence import (
    MIN_PASSES,
    MIN_SHARE,
    RADIUS_M,
    Spot,
    find_spots,
    load_spots,
    save_spots,
    spot_near,
)

LON, LAT = 50.30, 29.25
M_PER_DEG_LAT = 110_540.0


def north(metres: float) -> float:
    return LAT + metres / M_PER_DEG_LAT


def east(metres: float) -> float:
    return LON + metres / (111_320.0 * math.cos(math.radians(LAT)))


def seen(area, scenes, lon=LON, lat=LAT):
    return [(area, scene, lon, lat) for scene in scenes]


def scenes(n, first=1):
    return [f"S1A_IW_GRDH_1SDV_202401{day:02d}T023834_X" for day in range(first, first + n)]


def test_the_rule_is_the_one_fixed_before_any_result():
    assert (RADIUS_M, MIN_PASSES, MIN_SHARE) == (150.0, 3, 0.3)


def test_an_object_seen_on_enough_passes_is_a_spot():
    detections = seen("kharg", scenes(4)[:2]) + seen("kharg", scenes(4)[2:], east(40), north(-30))

    (spot,) = find_spots(detections, {"kharg": 10})

    assert (spot.area, spot.passes_seen, spot.passes_imaged) == ("kharg", 4, 10)
    assert spot.first == "2024-01-01" and spot.last == "2024-01-04"
    assert abs(spot.lon - LON) < 0.001 and abs(spot.lat - LAT) < 0.001


@pytest.mark.parametrize(
    ("passes_seen", "passes_imaged"),
    [(2, 4), (3, 11)],  # too few passes; 3 of 11 is under 30%
)
def test_too_few_passes_or_too_small_a_share_is_no_spot(passes_seen, passes_imaged):
    assert find_spots(seen("kharg", scenes(passes_seen)), {"kharg": passes_imaged}) == []


def test_several_detections_on_one_pass_count_once():
    one_pass = [("kharg", scenes(1)[0], east(d), LAT) for d in (0, 20, 40, 60, 80)]
    assert find_spots(one_pass, {"kharg": 5}) == []
    (spot,) = find_spots(one_pass + seen("kharg", scenes(3)[1:]), {"kharg": 5})
    assert spot.passes_seen == 3


def test_objects_farther_apart_than_the_radius_are_separate_spots():
    detections = seen("kharg", scenes(3)) + seen("kharg", scenes(3), east(400), LAT)
    assert len(find_spots(detections, {"kharg": 6})) == 2


def test_one_object_detected_with_jitter_gives_one_spot():
    jitter = [(0, 0), (60, 0), (0, 60), (-60, 0), (0, -60)]
    detections = [
        ("kharg", scene, east(dx), north(dy))
        for scene, (dx, dy) in zip(scenes(5), jitter, strict=True)
    ]
    assert len(find_spots(detections, {"kharg": 5})) == 1


def test_a_spot_is_found_near_a_position_within_the_radius_only():
    spot = Spot(LON, LAT, "kharg", 5, 10, "2024-01-01", "2025-11-03")
    assert spot_near(east(120), LAT, [spot]) == spot
    assert spot_near(east(200), LAT, [spot]) is None


def test_spots_are_saved_and_read_back_with_their_rule(tmp_path):
    spots = [Spot(LON, LAT, "kharg", 5, 10, "2024-01-01", "2025-11-03")]
    path = tmp_path / "spots.json"

    save_spots(path, spots, {"archive": "S1A before 2026-07-01"})
    meta, again = load_spots(path)

    assert again == spots
    assert meta["archive"] == "S1A before 2026-07-01"
    assert meta["rule"] == {"radius_m": 150.0, "min_passes": 3, "min_share": 0.3}
