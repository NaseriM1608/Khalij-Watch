"""Tests for turning Global Fishing Watch SAR presence rows into labels.

Rows are shaped as the 4Wings report returns them (checked against real
responses for the Fujairah anchorage, 2025-09 and 2026-09): cell centres on a
0.01 degree grid, an hourly ``date`` string, and vessel identity only on
AIS-matched rows.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.labeling.gfw import (
    AMBIGUOUS,
    GFW_ABSENT,
    GFW_UNMATCHED,
    LABELLED,
    Assignment,
    associate,
    cells_for_pass,
    label_fields,
    parse_rows,
)

PASS = dt.datetime(2025, 9, 17, 14, 17, 4, tzinfo=dt.UTC)
HOUR = "2025-09-17 14:00"


def matched_row(lat: float, lon: float, *, mmsi: str = "636020763", hour: str = HOUR) -> dict:
    return {
        "date": hour,
        "detections": 1,
        "lat": lat,
        "lon": lon,
        "mmsi": mmsi,
        "vessel_id": f"vid-{mmsi}",
        "ship_name": "STAR ELIZABETH",
        "flag": "LBR",
        "vessel_type": "CARGO",
        "gear_type": "CARGO",
        "imo": "9917488",
        "call_sign": "5LAP5",
        "report_dataset": "public-global-sar-presence:v4.0",
    }


def unmatched_row(lat: float, lon: float, *, detections: int = 1, hour: str = HOUR) -> dict:
    return {"date": hour, "detections": detections, "lat": lat, "lon": lon}


def cells(*rows_and_flags):
    parsed = []
    for rows, matched in rows_and_flags:
        parsed.extend(parse_rows(rows, matched=matched))
    return parsed


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_matched_rows_carry_vessel_identity():
    (cell,) = parse_rows([matched_row(25.25, 56.47999954223633)], matched=True)

    assert cell.matched
    assert cell.mmsi == 636020763
    assert cell.ship_name == "STAR ELIZABETH"
    assert cell.vessel_type == "CARGO"
    assert cell.hour == dt.datetime(2025, 9, 17, 14, tzinfo=dt.UTC)
    # Float noise in the reported centre does not change the cell.
    assert cell.key == (2525, 5648)


def test_unmatched_rows_have_no_identity():
    (cell,) = parse_rows([unmatched_row(25.25, 56.48, detections=2)], matched=False)

    assert not cell.matched
    assert cell.mmsi is None
    assert cell.detections == 2


def test_unusable_rows_are_dropped():
    rows = [
        {"date": HOUR, "detections": 1, "lat": None, "lon": 56.48},
        {"date": None, "detections": 1, "lat": 25.25, "lon": 56.48},
        unmatched_row(25.25, 56.48, detections=0),
    ]

    assert parse_rows(rows, matched=False) == []


def test_only_the_hours_around_the_pass_are_kept():
    parsed = cells(
        ([matched_row(25.25, 56.48), matched_row(25.20, 56.50, hour="2025-09-17 02:00")], True)
    )

    kept = cells_for_pass(parsed, PASS)

    assert [c.hour.hour for c in kept] == [14]


def test_a_pass_straddling_the_hour_keeps_both_hours():
    near_the_hour = dt.datetime(2025, 9, 17, 14, 59, 30, tzinfo=dt.UTC)
    parsed = cells(
        ([matched_row(25.25, 56.48), matched_row(25.20, 56.50, hour="2025-09-17 15:00")], True)
    )

    assert len(cells_for_pass(parsed, near_the_hour)) == 2


# --------------------------------------------------------------------------
# Association
# --------------------------------------------------------------------------


def test_one_matched_vessel_and_one_detection_in_a_cell_is_a_label():
    parsed = cells(([matched_row(25.25, 56.48)], True))

    (assignment,) = associate(parsed, [(56.4803, 25.2502)])

    assert assignment.status == LABELLED
    assert assignment.vessel.mmsi == 636020763


def test_a_detection_gfw_saw_but_could_not_match_is_flagged_not_labelled():
    parsed = cells(([unmatched_row(25.25, 56.48)], False))

    (assignment,) = associate(parsed, [(56.4803, 25.2502)])

    assert assignment.status == GFW_UNMATCHED
    assert assignment.vessel is None


def test_a_detection_gfw_did_not_see_is_absent():
    parsed = cells(([matched_row(25.25, 56.48)], True))

    (assignment,) = associate(parsed, [(56.5203, 25.2002)])

    assert assignment.status == GFW_ABSENT


@pytest.mark.parametrize(
    ("gfw", "ours"),
    [
        # GFW saw a matched and an unmatched object; which one is ours?
        (
            [([matched_row(25.25, 56.48)], True), ([unmatched_row(25.25, 56.48)], False)],
            [(56.4803, 25.2502)],
        ),
        # Two matched vessels share the cell.
        (
            [([matched_row(25.25, 56.48), matched_row(25.25, 56.48, mmsi="111")], True)],
            [(56.4803, 25.2502)],
        ),
    ],
)
def test_crowded_gfw_cells_are_ambiguous(gfw, ours):
    (assignment,) = associate(cells(*gfw), ours)

    assert assignment.status == AMBIGUOUS
    assert assignment.reason == "crowded_gfw"


def test_two_of_our_detections_in_one_cell_are_both_ambiguous():
    parsed = cells(([matched_row(25.25, 56.48)], True))

    assignments = associate(parsed, [(56.4803, 25.2502), (56.4782, 25.2521)])

    assert [a.status for a in assignments] == [AMBIGUOUS, AMBIGUOUS]
    assert {a.reason for a in assignments} == {"crowded_ours"}


def test_a_detection_near_an_edge_with_gfw_across_it_is_ambiguous():
    """Our georeferencing and GFW's differ by tens of metres, so a detection
    near the edge might be the one GFW placed in the neighbouring cell."""
    parsed = cells(([matched_row(25.25, 56.48)], True), ([unmatched_row(25.26, 56.48)], False))

    (assignment,) = associate(parsed, [(56.4803, 25.2548)])  # 0.0002 deg below the edge

    assert assignment.status == AMBIGUOUS
    assert assignment.reason == "edge"


def test_a_detection_near_an_edge_with_nothing_across_it_is_labelled():
    """Regression: a flat edge rule discarded about a third of every cell."""
    parsed = cells(([matched_row(25.25, 56.48)], True))

    (assignment,) = associate(parsed, [(56.4803, 25.2548)])

    assert assignment.status == LABELLED


def test_a_detection_near_a_corner_checks_the_diagonal_cell_too():
    parsed = cells(([matched_row(25.25, 56.48)], True), ([unmatched_row(25.26, 56.49)], False))

    (assignment,) = associate(parsed, [(56.4848, 25.2548)])

    assert assignment.status == AMBIGUOUS
    assert assignment.reason == "edge"


def test_a_labelled_detection_becomes_a_vessel_record_with_identity():
    (vessel,) = parse_rows([matched_row(25.25, 56.48)], matched=True)
    tanker = vessel.__class__(**{**vessel.__dict__, "gear_type": "TANKER"})

    fields = label_fields(Assignment(LABELLED, tanker))

    assert fields["kind"] == "vessel"
    assert fields["source"] == "gfw"
    assert fields["mmsi"] == 636020763
    assert fields["vessel_type"] == "cargo_tanker"
    assert fields["extra"]["ship_name"] == "STAR ELIZABETH"
    assert fields["extra"]["gfw_gear_type"] == "TANKER"


def test_an_unrecognised_gfw_type_keeps_its_raw_value_but_no_coarse_class():
    (vessel,) = parse_rows([matched_row(25.25, 56.48)], matched=True)
    odd = vessel.__class__(**{**vessel.__dict__, "gear_type": "NEW_KIND", "vessel_type": "OTHER"})

    fields = label_fields(Assignment(LABELLED, odd))

    assert fields["vessel_type"] is None
    assert fields["extra"]["gfw_gear_type"] == "NEW_KIND"


@pytest.mark.parametrize("status", [GFW_UNMATCHED, GFW_ABSENT, AMBIGUOUS])
def test_anything_short_of_a_label_stays_unlabelled_never_negative(status):
    """GFW failing to match a detection says nothing about whether it is a ship."""
    fields = label_fields(Assignment(status))

    assert fields["kind"] == "unlabelled"
    assert fields["mmsi"] is None
    assert fields["extra"]["gfw_status"] == status


def test_the_reason_for_ambiguity_is_kept():
    fields = label_fields(Assignment(AMBIGUOUS, reason="edge"))

    assert fields["extra"]["gfw_reason"] == "edge"


def test_cells_are_assigned_by_nearest_centre():
    parsed = cells(([matched_row(25.19, 56.48)], True), ([matched_row(25.20, 56.50)], True))

    below, above = associate(parsed, [(56.4800, 25.1930), (56.5000, 25.1970)])

    assert below.status == LABELLED and below.vessel.lat == pytest.approx(25.19)
    assert above.status == LABELLED and above.vessel.lat == pytest.approx(25.20)
