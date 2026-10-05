"""DARK_VESSEL events for one pass: our detections + GFW presence -> events with evidence cards.

The product path of Phase 6 v1 (PLAN §17, 2026-10-04 "Phase 6 feasibility"):
a stored pass is fused at the operating margin chosen on validation passes,
and every kept detection without an AIS candidate becomes a DARK_VESSEL event
carrying its evidence card. A pass whose presence feed is evidently
incomplete (core.fusion.guards) or empty gives no events: its dark calls would
be data gaps, not ships.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Sequence

from core.db.repository import NewEvent
from core.db.spatial import BBox
from core.fusion.assign import DARK, assign
from core.fusion.evidence import evidence_card
from core.fusion.gate import candidates
from core.fusion.guards import expected_ais, gfw_flag_for, is_gfw_artefact, is_presence_gap
from core.fusion.prefilter import SarDetection, prefilter
from core.fusion.presence import PresenceFix, hourly_tracks
from core.labeling.gfw import _floor_hour
from core.sar.coastline import LandPolygon
from core.sar.infrastructure import InfrastructureMask

DARK_VESSEL = "DARK_VESSEL"


def dark_vessel_events(
    *,
    scene: str,
    acquired: dt.datetime,
    detections: Sequence[tuple[int | None, SarDetection]],
    fixes: Sequence[PresenceFix],
    refs: Sequence[Sequence],
    land: Sequence[LandPolygon],
    infrastructure: InfrastructureMask,
    box: BBox,
    margin_m: float,
    margins: Sequence[float],
    r_over_v_s: float,
    shore_m: float,
    edge_m: float,
) -> tuple[list[NewEvent], dict]:
    """The pass's DARK_VESSEL events and a summary (counts and pass flags).

    ``detections`` pairs each detection with its stored id (or None);
    ``refs`` are GFW's SAR detections of the pass as (lon, lat, count, matched).
    """
    hour = _floor_hour(acquired)
    box_vessels = len({f.vessel_id for f in fixes if f.hour == hour and box.contains(f.lon, f.lat)})
    gfw_refs = sum(int(r[2]) for r in refs)
    gfw_matched = sum(int(r[2]) for r in refs if r[3])
    artefact = is_gfw_artefact(refs, box_vessels=box_vessels)
    expected = expected_ais(refs_total=gfw_refs, matched=gfw_matched, artefact=artefact)
    flags = {
        "gfw_artefact": artefact,
        "presence_gap": is_presence_gap(box_vessels=box_vessels, gfw_matched=expected),
        # Presence arrives about 96 h after a pass: an empty feed is missing data, not a dark sea.
        "no_presence": not fixes,
    }
    outcome = prefilter(
        [d for _, d in detections],
        land=land,
        infrastructure=infrastructure,
        box=box,
        shore_m=shore_m,
        edge_m=edge_m,
    )
    kept = [(i, d, p) for (i, d), p in zip(detections, outcome, strict=True) if p.kept]
    tracks = hourly_tracks(fixes, acquired)
    assigned = assign(
        candidates(
            [(d.lon, d.lat) for _, d, _ in kept], tracks, r_over_v_s=r_over_v_s, margin_m=margin_m
        )
    )
    dark = [(i, d, p) for (i, d, p), a in zip(kept, assigned, strict=True) if a.status == DARK]
    summary = {
        "detections": len(detections),
        "prefilter": dict(sorted(Counter(p.reason for p in outcome if p.reason).items())),
        "kept": len(kept),
        "dark": len(dark),
        "tracks": len(tracks),
        "box_vessels": box_vessels,
        "gfw_matched": gfw_matched,
        **flags,
    }
    if flags["presence_gap"] or flags["no_presence"]:
        return [], summary
    datasets = sorted({f.report_dataset for f in fixes if f.report_dataset})
    events = []
    for detection_id, detection, prefiltered in dark:
        card = evidence_card(
            scene=scene,
            acquired=acquired,
            detection=detection,
            prefiltered=prefiltered,
            tracks=tracks,
            margin_m=margin_m,
            margins=margins,
            r_over_v_s=r_over_v_s,
            gfw_matched=gfw_flag_for(detection.lon, detection.lat, refs),
            pass_flags=flags,
            datasets=datasets,
        )
        events.append(
            NewEvent(
                event_type=DARK_VESSEL,
                ts=acquired,
                lon=detection.lon,
                lat=detection.lat,
                severity=card["confidence"],
                detection_id=detection_id,
                evidence=card,
            )
        )
    return events, summary
