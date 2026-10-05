"""DARK_VESSEL events for one stored pass: detections + presence -> events with evidence.

The product path of Phase 6 v1: a pass the console already holds is fused at
the operating margin chosen on validation passes, and every kept detection
with no AIS candidate becomes a DARK_VESSEL event carrying its evidence card.
A pass whose presence feed is evidently incomplete gives no events at all,
since its dark calls would be data gaps, not ships (PLAN §17, 2026-10-04
"Phase 6 feasibility").
"""

from __future__ import annotations

import datetime as dt

from core.db.spatial import BBox
from core.fusion.events import dark_vessel_events
from core.fusion.prefilter import SarDetection
from core.fusion.presence import PresenceFix
from core.sar.infrastructure import InfrastructureMask

ACQ = dt.datetime(2026, 9, 25, 14, 15, 51, tzinfo=dt.UTC)
BOX = BBox(56.43, 25.13, 56.63, 25.31)
MARGINS = (0.0, 500.0, 2000.0)


def anchored(vessel_id, lon, lat):
    return [
        PresenceFix(
            vessel_id=vessel_id,
            hour=dt.datetime(2026, 9, 25, h, tzinfo=dt.UTC),
            lon=lon,
            lat=lat,
            report_dataset="public-global-presence:v4.0",
        )
        for h in (13, 14, 15)
    ]


def run(detections, fixes, refs=()):
    return dark_vessel_events(
        scene="S1C_TEST.SAFE",
        acquired=ACQ,
        detections=detections,
        fixes=fixes,
        refs=list(refs),
        land=[],
        infrastructure=InfrastructureMask([], radius_m=300.0),
        box=BOX,
        margin_m=500.0,
        margins=MARGINS,
        r_over_v_s=118.0,
        shore_m=1000.0,
        edge_m=500.0,
    )


def test_only_kept_detections_without_an_ais_candidate_become_dark_vessel_events():
    detections = [
        (11, SarDetection(56.502, 25.201, 0.95)),  # at an anchored AIS vessel
        (12, SarDetection(56.58, 25.27, 0.90)),  # about 11 km from it: dark
        (13, SarDetection(56.432, 25.20, 0.90)),  # at the box edge: set aside
    ]
    events, summary = run(detections, anchored("a", 56.50, 25.20), refs=[(56.58, 25.27, 1, False)])
    (event,) = events
    assert (event.event_type, event.detection_id, event.lon, event.lat) == (
        "DARK_VESSEL",
        12,
        56.58,
        25.27,
    )
    assert event.ts == ACQ
    assert event.severity == event.evidence["confidence"] == "high"
    assert event.evidence["gfw"]["corroborates"] is True
    assert event.evidence["datasets"] == ["public-global-presence:v4.0"]
    assert summary["kept"] == 2 and summary["dark"] == 1 and summary["prefilter"] == {"edge": 1}


def test_a_pass_with_a_presence_gap_gives_no_events():
    refs = [
        (56.44 + 0.01 * i, 25.15, 1, True) for i in range(12)
    ]  # GFW matched 12, presence holds 1 vessel
    events, summary = run(
        [(12, SarDetection(56.58, 25.27, 0.9))], anchored("a", 56.50, 25.20), refs=refs
    )
    assert events == []
    assert summary["presence_gap"] is True


def test_the_pass_summary_reports_the_flags_shown_on_every_card():
    events, summary = run([(12, SarDetection(56.58, 25.27, 0.9))], anchored("a", 56.50, 25.20))
    assert summary["gfw_artefact"] is False and summary["presence_gap"] is False
    assert events[0].evidence["pass_flags"] == {
        "gfw_artefact": False,
        "presence_gap": False,
        "no_presence": False,
        "presence_checked": False,
        "gfw_refs": 0,
    }


def test_a_pass_without_any_presence_gives_no_events():
    # Presence arrives about 96 h after a pass; before that the feed is empty
    # and every detection would look dark.
    events, summary = run([(12, SarDetection(56.58, 25.27, 0.9))], [])
    assert events == []
    assert summary["no_presence"] is True


def test_on_a_gfw_artefact_pass_gfws_unmatched_flag_corroborates_nothing():
    # GFW matched none of 12 detections although 20+ AIS vessels sit in the box.
    fixes = [f for i in range(25) for f in anchored(f"v{i}", 56.45 + 0.004 * i, 25.15)]
    refs = [(56.44 + 0.01 * i, 25.16, 1, False) for i in range(11)] + [(56.58, 25.27, 1, False)]
    events, summary = run([(12, SarDetection(56.58, 25.27, 0.9))], fixes, refs=refs)
    assert summary["gfw_artefact"] is True
    assert events[0].evidence["gfw"] == {
        "cell_matched": None,
        "corroborates": None,
        "cell_state": "untrusted",
    }


def test_cards_state_gfws_cell_verdict_and_whether_presence_could_be_checked():
    refs = [(56.58, 25.27, 1, True), (56.58, 25.27, 1, False)]  # mixed cell
    events, summary = run(
        [(12, SarDetection(56.58, 25.27, 0.9))], anchored("a", 56.50, 25.20), refs=refs
    )
    card = events[0].evidence
    assert card["gfw"]["cell_state"] == "mixed"
    assert card["pass_flags"]["gfw_refs"] == 2
    assert card["pass_flags"]["presence_checked"] is False  # 1 GFW match: too few to judge
    assert summary["presence_checked"] is False


def test_a_pass_where_gfw_published_nothing_says_so_on_the_card():
    events, _ = run([(12, SarDetection(56.58, 25.27, 0.9))], anchored("a", 56.50, 25.20), refs=[])
    assert events[0].evidence["gfw"]["cell_state"] == "no_gfw_data"
