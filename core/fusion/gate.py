"""The cell-hour gate between our SAR detections and GFW hourly AIS presence.

Without position-level AIS (GFW's tracks endpoint is 403 for our token) a
vessel is the polyline through its hourly cell centres around the pass. One
of our detections is a candidate for that vessel when it lies within the
vessel's gate of the polyline:

- the cell half-diagonal (about 750 m in the Gulf), since the true position
  can be anywhere in the cell;
- an azimuth allowance: a target with radial velocity v_r is displaced along
  track by about (R/V) * v_r in the image (PLAN §8), with R/V about 105-118 s
  on Sentinel-1 IW. Heading within the hour is unknown, so the allowance takes
  the vessel's whole speed (its fastest hourly cell displacement): an
  allowance, not a correction. R/V comes from the scene and is passed in,
  never hardcoded here;
- a margin, over which the dark rate is reported as a curve.

A detection with no candidate is dark (PLAN §17, 2026-10-04 "Phase 6
feasibility"). On the busy Fujairah pass of 2026-09-25, GFW matched 65
vessels with an id; 58 were in the presence window, and GFW's own SAR position
for 53 of those 58 lay within 0.8 km of the vessel's hourly polyline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from core.fusion.geometry import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR, distance_to_polyline_m
from core.fusion.presence import HourlyTrack
from core.labeling.gfw import CELL_DEG


@dataclass(frozen=True)
class Candidate:
    """An AIS vessel whose hourly track passes within its gate of a detection."""

    vessel_id: str
    distance_m: float
    gate_m: float

    @property
    def score(self) -> float:
        """Distance as a fraction of the gate: 0 on the track, 1 at the gate's edge."""
        return self.distance_m / self.gate_m


def cell_half_diagonal_m(lat: float) -> float:
    """Half the diagonal of a 0.01 degree cell at latitude ``lat``."""
    return 0.5 * math.hypot(
        CELL_DEG * M_PER_DEG_LAT, CELL_DEG * M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat))
    )


def gate_m(track: HourlyTrack, *, r_over_v_s: float, margin_m: float) -> float:
    """Gate radius around a vessel's hourly polyline: cell floor + azimuth allowance + margin.

    A sparse (single-fix) track has unknown speed and gets no allowance.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (r_over_v_s, margin_m)):
        raise ValueError(
            f"r_over_v_s {r_over_v_s} and margin_m {margin_m} must be finite and non-negative"
        )
    lat = sum(f.lat for f in track.fixes) / len(track.fixes)
    return cell_half_diagonal_m(lat) + r_over_v_s * (track.speed_mps or 0.0) + margin_m


def candidates(
    positions: Sequence[tuple[float, float]],
    tracks: Iterable[HourlyTrack],
    *,
    r_over_v_s: float,
    margin_m: float,
) -> list[list[Candidate]]:
    """For each (lon, lat) detection, the vessels whose gate it falls in, best (lowest score) first."""
    gated = [
        (t.vessel_id, t.gate_vertices, gate_m(t, r_over_v_s=r_over_v_s, margin_m=margin_m))
        for t in tracks
    ]
    found: list[list[Candidate]] = []
    for lon, lat in positions:
        inside = [
            Candidate(vessel_id=vessel_id, distance_m=distance, gate_m=gate)
            for vessel_id, vertices, gate in gated
            if (distance := distance_to_polyline_m(lon, lat, vertices)) <= gate
        ]
        found.append(sorted(inside, key=lambda c: (c.score, c.vessel_id)))
    return found
