"""The cell-hour gate: which AIS vessels could be one of our SAR detections.

v1 has no position-level AIS, so a vessel is a polyline through its hourly
cell centres (H-1, H, H+1). A detection is a candidate for a vessel when it
lies within the gate of that polyline: the cell half-diagonal, plus an
azimuth allowance (R/V times a speed bounded by the cell displacement), plus
a margin that the dark rate is swept over. A detection with no candidate is
dark (PLAN §17, 2026-10-04 "Phase 6 feasibility").
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from core.fusion.gate import (
    Candidate,
    candidates,
    cell_half_diagonal_m,
    distance_to_polyline_m,
    gate_m,
)
from core.fusion.presence import PresenceFix, hourly_tracks

PASS = dt.datetime(2026, 9, 25, 14, 15, 51, tzinfo=dt.UTC)
M_PER_DEG_LAT = 110_540.0  # the local equirectangular scales of core.geo.linear_deviation_m
M_PER_DEG_LON_EQ = 111_320.0


def track(vessel_id="v1", cells=((13, 56.50, 25.20), (14, 56.50, 25.20), (15, 56.50, 25.20))):
    fixes = [
        PresenceFix(
            vessel_id=vessel_id, hour=dt.datetime(2026, 9, 25, h, tzinfo=dt.UTC), lon=lon, lat=lat
        )
        for h, lon, lat in cells
    ]
    (built,) = hourly_tracks(fixes, PASS)
    return built


# ---------------------------------------------------------------------------
# Distance to a polyline
# ---------------------------------------------------------------------------


def test_a_point_on_a_segment_is_at_distance_zero():
    assert distance_to_polyline_m(56.55, 25.0, [(56.50, 25.0), (56.60, 25.0)]) == pytest.approx(
        0.0, abs=1e-6
    )


def test_distance_to_a_segment_is_perpendicular_inside_it():
    d = distance_to_polyline_m(56.55, 25.01, [(56.50, 25.0), (56.60, 25.0)])
    assert d == pytest.approx(0.01 * M_PER_DEG_LAT, rel=1e-3)


def test_distance_beyond_a_segment_end_is_to_the_end_point():
    d = distance_to_polyline_m(56.63, 25.0, [(56.50, 25.0), (56.60, 25.0)])
    assert d == pytest.approx(0.03 * M_PER_DEG_LON_EQ * math.cos(math.radians(25.0)), rel=1e-3)


def test_distance_to_a_single_vertex_is_the_point_distance():
    d = distance_to_polyline_m(56.50, 25.02, [(56.50, 25.0)])
    assert d == pytest.approx(0.02 * M_PER_DEG_LAT, rel=1e-3)


def test_distance_to_a_bent_polyline_is_to_its_nearest_segment():
    bent = [(56.50, 25.0), (56.60, 25.0), (56.60, 25.10)]
    assert distance_to_polyline_m(56.61, 25.05, bent) == pytest.approx(
        0.01 * M_PER_DEG_LON_EQ * math.cos(math.radians(25.05)), rel=1e-3
    )


def test_an_empty_polyline_is_rejected():
    with pytest.raises(ValueError, match="vertex"):
        distance_to_polyline_m(56.5, 25.0, [])


# ---------------------------------------------------------------------------
# Gate size
# ---------------------------------------------------------------------------


def test_cell_half_diagonal_at_the_gulf_is_about_750_m():
    expected = 0.5 * math.hypot(
        0.01 * M_PER_DEG_LAT, 0.01 * M_PER_DEG_LON_EQ * math.cos(math.radians(25.0))
    )
    assert cell_half_diagonal_m(25.0) == pytest.approx(expected)
    assert 740 < cell_half_diagonal_m(25.0) < 760


def test_a_stationary_vessel_gets_the_cell_floor_plus_the_margin():
    t = track()
    assert gate_m(t, r_over_v_s=110.0, margin_m=0.0) == pytest.approx(cell_half_diagonal_m(25.20))
    assert gate_m(t, r_over_v_s=110.0, margin_m=500.0) == pytest.approx(
        cell_half_diagonal_m(25.20) + 500.0
    )


def test_a_moving_vessel_gets_an_azimuth_allowance_of_r_over_v_times_its_speed():
    moving = track(cells=((13, 56.50, 25.0), (14, 56.50, 25.05), (15, 56.50, 25.10)))
    speed = 0.05 * M_PER_DEG_LAT / 3600
    floor = cell_half_diagonal_m(25.05)
    assert gate_m(moving, r_over_v_s=110.0, margin_m=0.0) == pytest.approx(
        floor + 110.0 * speed, rel=1e-3
    )


def test_a_sparse_track_gets_no_allowance_because_its_speed_is_unknown():
    sparse = track(cells=((14, 56.50, 25.20),))
    assert gate_m(sparse, r_over_v_s=110.0, margin_m=100.0) == pytest.approx(
        cell_half_diagonal_m(25.20) + 100.0
    )


def test_gate_rejects_a_negative_margin_or_r_over_v():
    with pytest.raises(ValueError):
        gate_m(track(), r_over_v_s=110.0, margin_m=-1.0)
    with pytest.raises(ValueError):
        gate_m(track(), r_over_v_s=-5.0, margin_m=0.0)


# ---------------------------------------------------------------------------
# Candidates per detection
# ---------------------------------------------------------------------------


def test_a_detection_in_a_stationary_vessels_cell_has_it_as_candidate():
    (found,) = candidates([(56.502, 25.201)], [track()], r_over_v_s=110.0, margin_m=0.0)
    assert [c.vessel_id for c in found] == ["v1"]
    assert found[0].distance_m < found[0].gate_m


def test_a_detection_far_from_every_vessel_is_dark():
    (found,) = candidates(
        [(56.50, 25.25)], [track()], r_over_v_s=110.0, margin_m=0.0
    )  # about 5.5 km north
    assert found == []


def test_a_detection_between_hourly_cells_matches_the_moving_vessel():
    moving = track("m", cells=((13, 56.50, 25.0), (14, 56.50, 25.05), (15, 56.50, 25.10)))
    (found,) = candidates([(56.501, 25.075)], [moving], r_over_v_s=110.0, margin_m=0.0)
    assert [c.vessel_id for c in found] == ["m"]


def test_the_margin_turns_a_near_miss_into_a_candidate():
    position = [(56.50, 25.20 + 1200 / M_PER_DEG_LAT)]  # 1.2 km north of the cell centre
    assert candidates(position, [track()], r_over_v_s=110.0, margin_m=0.0) == [[]]
    (found,) = candidates(position, [track()], r_over_v_s=110.0, margin_m=500.0)
    assert [c.vessel_id for c in found] == ["v1"]


def test_candidates_are_ordered_by_distance_relative_to_their_gate():
    near = track("near", cells=((13, 56.50, 25.20), (14, 56.50, 25.20), (15, 56.50, 25.20)))
    far = track("far", cells=((13, 56.51, 25.20), (14, 56.51, 25.20), (15, 56.51, 25.20)))
    (found,) = candidates([(56.503, 25.20)], [far, near], r_over_v_s=110.0, margin_m=500.0)
    assert [c.vessel_id for c in found] == ["near", "far"]
    assert all(isinstance(c, Candidate) for c in found)


def test_results_are_aligned_with_the_detections_given():
    found = candidates([(56.50, 25.25), (56.50, 25.20)], [track()], r_over_v_s=110.0, margin_m=0.0)
    assert [len(f) for f in found] == [0, 1]


# ---------------------------------------------------------------------------
# Tracks that do not span the pass time
# ---------------------------------------------------------------------------
# A fix in hour H lies somewhere in [H, H+1); the pass is at H + f (f = 0.264
# here). A vessel seen only at H-1 and H can be up to f of an hour of travel
# beyond its H cell at the pass, and one seen only at H and H+1 up to (1 - f)
# of an hour before it, so such tracks are extended along their direction.

F = (15 * 60 + 51) / 3600  # PASS is 14:15:51


def test_a_track_seen_at_h_minus_1_and_h_is_extended_forward_by_the_pass_fraction():
    moving = track(cells=((13, 56.50, 25.00), (14, 56.50, 25.10)))
    ahead = 25.10 + 0.8 * F * 0.10  # within the extension, 1.5 km beyond the H cell
    assert candidates([(56.50, ahead)], [moving], r_over_v_s=0.0, margin_m=0.0) != [[]]
    beyond = 25.10 + F * 0.10 + 0.02  # past the extension by 2.2 km
    assert candidates([(56.50, beyond)], [moving], r_over_v_s=0.0, margin_m=0.0) == [[]]


def test_a_track_seen_at_h_and_h_plus_1_is_extended_backward_by_the_rest_of_the_hour():
    moving = track(cells=((14, 56.50, 25.10), (15, 56.50, 25.20)))
    behind = 25.10 - 0.8 * (1 - F) * 0.10  # 6.5 km before the H cell
    assert candidates([(56.50, behind)], [moving], r_over_v_s=0.0, margin_m=0.0) != [[]]


def test_a_track_spanning_the_pass_is_not_extended():
    full = track(cells=((13, 56.50, 25.00), (14, 56.50, 25.10), (15, 56.50, 25.20)))
    assert full.gate_vertices == full.vertices


def test_extending_a_vessel_that_stays_in_one_cell_changes_nothing():
    still = track(cells=((13, 56.50, 25.10), (14, 56.50, 25.10)))
    assert {tuple(v) for v in still.gate_vertices} == {(56.50, 25.10)}


def test_gate_rejects_nan_or_infinite_parameters():
    with pytest.raises(ValueError):
        gate_m(track(), r_over_v_s=float("nan"), margin_m=0.0)
    with pytest.raises(ValueError):
        gate_m(track(), r_over_v_s=110.0, margin_m=float("inf"))
