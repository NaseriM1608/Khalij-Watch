"""Evidence cards for DARK_VESSEL events: why a detection is called dark.

PLAN §9 and §15: never surface an alert that cannot be explained. A card
records what the analyst needs to audit a dark call: the nearest AIS tracks
and how far outside their gates the detection lies, how far it stays dark as
the gate margin grows (the confidence tier), GFW's own matched flag for the
cell, the pre-filter distances, and the dataset versions (PLAN §17,
2026-10-04 "Phase 6 feasibility").
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.fusion.evidence import confidence_tier, dark_up_to_m, evidence_card, nearest_tracks
from core.fusion.gate import cell_half_diagonal_m
from core.fusion.prefilter import Prefiltered, SarDetection
from core.fusion.presence import PresenceFix, hourly_tracks

PASS = dt.datetime(2026, 9, 25, 14, 15, 51, tzinfo=dt.UTC)
MARGINS = (0.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)
M_PER_DEG_LAT = 110_540.0


def anchored(vessel_id, lon, lat, **identity):
    fixes = [
        PresenceFix(
            vessel_id=vessel_id,
            hour=dt.datetime(2026, 9, 25, h, tzinfo=dt.UTC),
            lon=lon,
            lat=lat,
            **identity,
        )
        for h in (13, 14, 15)
    ]
    (track,) = hourly_tracks(fixes, PASS)
    return track


def north(metres: float) -> float:
    return metres / M_PER_DEG_LAT


# ---------------------------------------------------------------------------
# How far a detection stays dark
# ---------------------------------------------------------------------------


def test_a_detection_stays_dark_up_to_the_largest_margin_below_its_gap():
    floor = cell_half_diagonal_m(25.20)
    tracks = [anchored("a", 56.50, 25.20)]
    lat = 25.20 + north(floor + 1200)  # 1.2 km outside the margin-0 gate
    assert dark_up_to_m(56.50, lat, tracks, margins=MARGINS, r_over_v_s=118.0) == 1000.0


def test_a_detection_inside_a_gate_at_margin_zero_is_not_dark_at_all():
    tracks = [anchored("a", 56.50, 25.20)]
    assert dark_up_to_m(56.50, 25.20, tracks, margins=MARGINS, r_over_v_s=118.0) is None


def test_a_detection_with_no_ais_vessel_at_all_stays_dark_at_every_margin():
    assert dark_up_to_m(56.50, 25.20, [], margins=MARGINS, r_over_v_s=118.0) == 5000.0


@pytest.mark.parametrize(
    ("margin", "tier"),
    [
        (5000.0, "high"),
        (2000.0, "high"),
        (1000.0, "medium"),
        (500.0, "medium"),
        (250.0, "low"),
        (0.0, "low"),
    ],
)
def test_the_confidence_tier_follows_how_far_the_detection_stays_dark(margin, tier):
    assert confidence_tier(margin) == tier


# ---------------------------------------------------------------------------
# Nearest tracks and the card
# ---------------------------------------------------------------------------


def test_nearest_tracks_list_identity_hourly_cells_and_distance_nearest_first():
    near = anchored("near", 56.50, 25.21, mmsi=538005255, ship_name="ZENITH FAITH", flag="MHL")
    far = anchored("far", 56.50, 25.25)
    listed = nearest_tracks(56.50, 25.20, [far, near], n=1)
    (only,) = listed
    assert only["vessel_id"] == "near"
    assert (
        only["mmsi"] == 538005255 and only["ship_name"] == "ZENITH FAITH" and only["flag"] == "MHL"
    )
    assert only["distance_m"] == pytest.approx(0.01 * M_PER_DEG_LAT, rel=1e-3)
    assert [c["hour"] for c in only["cells"]] == [
        "2026-09-25T13:00:00+00:00",
        "2026-09-25T14:00:00+00:00",
        "2026-09-25T15:00:00+00:00",
    ]


def test_a_card_explains_a_dark_call():
    floor = cell_half_diagonal_m(25.20)
    tracks = [anchored("a", 56.50, 25.20)]
    detection = SarDetection(lon=56.50, lat=25.20 + north(floor + 2500), score=0.93)
    card = evidence_card(
        scene="S1C_TEST.SAFE",
        acquired=PASS,
        detection=detection,
        prefiltered=Prefiltered(reason=None, merged_into=None, shore_m=4200.0, edge_m=3100.0),
        tracks=tracks,
        margin_m=500.0,
        margins=MARGINS,
        r_over_v_s=118.0,
        gfw_matched=False,
        gfw_cell_state="unmatched",
        pass_flags={"gfw_artefact": False, "presence_gap": False},
        datasets=["public-global-presence:v4.0"],
    )
    assert card["status"] == "dark"
    assert card["scene"] == "S1C_TEST.SAFE" and card["acquired"] == PASS.isoformat()
    assert card["detection"] == {"lon": detection.lon, "lat": detection.lat, "score": 0.93}
    assert card["gate"]["margin_m"] == 500.0 and card["gate"]["r_over_v_s"] == 118.0
    assert card["dark_up_to_m"] == 2000.0 and card["confidence"] == "high"
    assert card["gfw"] == {"cell_matched": False, "corroborates": True, "cell_state": "unmatched"}
    assert card["prefilter"] == {"shore_m": 4200.0, "edge_m": 3100.0}
    assert card["nearest_ais"][0]["vessel_id"] == "a"
    assert card["datasets"] == ["public-global-presence:v4.0"]
    assert card["pass_flags"] == {"gfw_artefact": False, "presence_gap": False}
    assert card["chip"] == {"scene": "S1C_TEST.SAFE", "lon": detection.lon, "lat": detection.lat}
    assert "Powered by Global Fishing Watch" in card["attribution"]


def test_a_card_refuses_a_detection_that_is_not_dark_at_the_operating_margin():
    tracks = [anchored("a", 56.50, 25.20)]
    with pytest.raises(ValueError, match="not dark"):
        evidence_card(
            scene="S",
            acquired=PASS,
            detection=SarDetection(lon=56.50, lat=25.20, score=0.9),
            prefiltered=Prefiltered(reason=None, merged_into=None, shore_m=5000.0, edge_m=5000.0),
            tracks=tracks,
            margin_m=0.0,
            margins=MARGINS,
            r_over_v_s=118.0,
            gfw_matched=None,
            pass_flags={},
            datasets=[],
        )


def test_a_card_never_holds_infinite_distances_so_it_stays_valid_json():
    # No AIS vessel at all and no land loaded: both distances are infinite in
    # the computation; the API serialises with allow_nan=False.
    import json

    card = evidence_card(
        scene="S",
        acquired=PASS,
        detection=SarDetection(lon=56.50, lat=25.20, score=0.9),
        prefiltered=Prefiltered(reason=None, merged_into=None, shore_m=float("inf"), edge_m=3000.0),
        tracks=[],
        margin_m=0.0,
        margins=MARGINS,
        r_over_v_s=118.0,
        gfw_matched=None,
        pass_flags={},
        datasets=[],
    )
    assert card["gate"]["nearest_gap_m"] is None
    assert card["prefilter"]["shore_m"] is None
    json.dumps(card, allow_nan=False)


def test_a_card_names_the_detector_and_draws_each_nearest_vessels_gate_track():
    tracks = [anchored("a", 56.50, 25.20)]
    card = evidence_card(
        scene="S",
        acquired=PASS,
        detection=SarDetection(
            lon=56.50, lat=25.25, score=1.0, detector="cfar", detector_version="L4.4"
        ),
        prefiltered=Prefiltered(reason=None, merged_into=None, shore_m=5000.0, edge_m=5000.0),
        tracks=tracks,
        margin_m=0.0,
        margins=MARGINS,
        r_over_v_s=118.0,
        gfw_matched=None,
        gfw_cell_state="mixed",
        pass_flags={},
        datasets=[],
    )
    assert card["detector"] == {"name": "cfar", "version": "L4.4"}
    assert card["gfw"]["cell_state"] == "mixed"
    assert card["nearest_ais"][0]["track"] == [[56.50, 25.20]] * 3
