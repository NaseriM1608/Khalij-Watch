"""Evidence cards for DARK_VESSEL events: why a detection is called dark.

Never surface an alert that cannot be explained (PLAN §9, §15). A dark call in
cell-hour fusion rests on the gate, so the card records the nearest AIS
vessels' hourly tracks and how far the detection lies outside their gates,
how far it stays dark as the gate margin grows (which sets the confidence
tier: a detection dark even with a 2 km margin is far from any AIS track),
GFW's own matched flag for the detection's cell, the pre-filter distances,
the pass flags and the dataset versions (PLAN §17, 2026-10-04 "Phase 6
feasibility"). The SAR chip is referenced by scene and position, to be cut
when shown.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence

from core.fusion.assign import DARK
from core.fusion.gate import gate_m
from core.fusion.geometry import distance_to_polyline_m
from core.fusion.prefilter import Prefiltered, SarDetection
from core.fusion.presence import HourlyTrack

# Confidence tiers by the largest swept margin at which a detection stays dark.
TIER_HIGH_M = 2_000.0
TIER_MEDIUM_M = 500.0
NEAREST_LISTED = 3
ATTRIBUTION = (
    "AIS presence and SAR vessel detections: Powered by Global Fishing Watch (CC BY-NC 4.0). "
    "Contains modified Copernicus Sentinel data."
)


def _gap_m(lon: float, lat: float, track: HourlyTrack, r_over_v_s: float) -> float:
    """How far outside the vessel's margin-0 gate the position lies (negative inside it)."""
    distance = distance_to_polyline_m(lon, lat, track.gate_vertices)
    return distance - gate_m(track, r_over_v_s=r_over_v_s, margin_m=0.0)


def _nearest_gap_m(
    lon: float, lat: float, tracks: Sequence[HourlyTrack], r_over_v_s: float
) -> float:
    return min((_gap_m(lon, lat, t, r_over_v_s) for t in tracks), default=math.inf)


def dark_up_to_m(
    lon: float,
    lat: float,
    tracks: Sequence[HourlyTrack],
    *,
    margins: Sequence[float],
    r_over_v_s: float,
) -> float | None:
    """The largest swept margin at which the position is still dark; None if it is not dark at all."""
    gap = _nearest_gap_m(lon, lat, tracks, r_over_v_s)
    dark = [m for m in sorted(margins) if m < gap]
    return dark[-1] if dark else None


def confidence_tier(dark_up_to: float) -> str:
    if dark_up_to >= TIER_HIGH_M:
        return "high"
    return "medium" if dark_up_to >= TIER_MEDIUM_M else "low"


def nearest_tracks(
    lon: float, lat: float, tracks: Sequence[HourlyTrack], n: int = NEAREST_LISTED
) -> list[dict]:
    """The ``n`` AIS vessels whose hourly tracks pass nearest, with identity and cells."""
    ranked = sorted(tracks, key=lambda t: distance_to_polyline_m(lon, lat, t.gate_vertices))[:n]
    listed = []
    for t in ranked:
        named = next((f for f in t.fixes if f.mmsi or f.ship_name), t.fixes[0])
        listed.append(
            {
                "vessel_id": t.vessel_id,
                "mmsi": named.mmsi,
                "imo": named.imo,
                "ship_name": named.ship_name,
                "flag": named.flag,
                "vessel_type": named.vessel_type,
                "distance_m": round(distance_to_polyline_m(lon, lat, t.gate_vertices), 1),
                "stationary": t.stationary,
                "sparse": t.sparse,
                "cells": [
                    {"hour": f.hour.isoformat(), "lon": f.lon, "lat": f.lat} for f in t.fixes
                ],
            }
        )
    return listed


def evidence_card(
    *,
    scene: str,
    acquired: dt.datetime,
    detection: SarDetection,
    prefiltered: Prefiltered,
    tracks: Sequence[HourlyTrack],
    margin_m: float,
    margins: Sequence[float],
    r_over_v_s: float,
    gfw_matched: bool | None,
    pass_flags: dict,
    datasets: Sequence[str],
) -> dict:
    """The card of a detection that is dark at the operating ``margin_m``."""
    gap = _nearest_gap_m(detection.lon, detection.lat, tracks, r_over_v_s)
    if not margin_m < gap:
        raise ValueError(
            f"detection at {detection.lon}, {detection.lat} is not dark at margin {margin_m} m"
        )
    up_to = dark_up_to_m(
        detection.lon, detection.lat, tracks, margins=margins, r_over_v_s=r_over_v_s
    )
    up_to = margin_m if up_to is None else max(up_to, margin_m)
    return {
        "status": DARK,
        "scene": scene,
        "acquired": acquired.isoformat(),
        "detection": {"lon": detection.lon, "lat": detection.lat, "score": detection.score},
        "gate": {"margin_m": margin_m, "r_over_v_s": r_over_v_s, "nearest_gap_m": round(gap, 1)},
        "dark_up_to_m": up_to,
        "confidence": confidence_tier(up_to),
        "gfw": {
            "cell_matched": gfw_matched,
            "corroborates": None if gfw_matched is None else not gfw_matched,
        },
        "prefilter": {"shore_m": prefiltered.shore_m, "edge_m": prefiltered.edge_m},
        "nearest_ais": nearest_tracks(detection.lon, detection.lat, tracks),
        "pass_flags": dict(pass_flags),
        "datasets": list(datasets),
        "chip": {"scene": scene, "lon": detection.lon, "lat": detection.lat},
        "attribution": ATTRIBUTION,
    }
