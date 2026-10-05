"""Pre-filters on our SAR detections before they enter the dark-rate count.

A detection counts only if it is one vessel in open, imaged water (PLAN §17,
2026-10-04 "Phase 6 feasibility", "An honest dark rate"):

- duplicates are merged first: the product model leaves 2.8% of peaks within
  100 m of another (tile seams). The 100 m radius does not merge CFAR
  fragments of one large ship, which lie about 170 m apart (PLAN §17,
  2026-09-27); the dark rate uses the model's detections;
- detections within ``shore_m`` of land (GFW's dark-vessel convention is
  1 km), on fixed infrastructure (the time-aware OPD mask, applied for the pass
  date by the caller), or within ``edge_m`` of the processed box are set aside
  with a reason, never silently dropped. Shore distances are exact only within
  the land the caller loaded around the box.

Not yet here: azimuth ghosts, which need the per-sub-swath offset from the
annotation XML (not cached for stored passes; none were seen at 10 m on S1A,
PLAN §17, 2026-09-29), and the scene footprint's no-data edge, which the
comparison results do not store (only exact-zero pixels were masked upstream).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.db.spatial import BBox
from core.fusion.geometry import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR, local_distance_m
from core.sar.coastline import LandPolygon
from core.sar.infrastructure import InfrastructureMask

DUPLICATE = "duplicate"
SHORE = "shore"
INFRASTRUCTURE = "infrastructure"
EDGE = "edge"

# PLAN §17 v1: duplicate peaks and fragments of one ship are merged at about 100 m.
DUPLICATE_RADIUS_M = 100.0


@dataclass(frozen=True)
class SarDetection:
    """One of our detections: position and score (model probability; CFAR detections use 1.0)."""

    lon: float
    lat: float
    score: float


@dataclass(frozen=True)
class Prefiltered:
    """A detection's pre-filter outcome, with the distances measured for its evidence card."""

    reason: str | None
    merged_into: int | None
    shore_m: float
    edge_m: float

    @property
    def kept(self) -> bool:
        return self.reason is None


def merge_duplicates(
    detections: Sequence[SarDetection], radius_m: float = DUPLICATE_RADIUS_M
) -> list[int]:
    """For each detection, the index of the detection it is kept as: itself, or a stronger one within ``radius_m``.

    Strongest first (ties in input order); a merged detection never absorbs
    others, so merging does not chain beyond the radius.
    """
    keeper = list(range(len(detections)))
    kept: list[int] = []
    for i in sorted(range(len(detections)), key=lambda j: -detections[j].score):
        d = detections[i]
        near = next(
            (
                k
                for k in kept
                if local_distance_m(d.lon, d.lat, detections[k].lon, detections[k].lat) <= radius_m
            ),
            None,
        )
        if near is None:
            kept.append(i)
        else:
            keeper[i] = near
    return keeper


def distance_to_land_m(lon: float, lat: float, polygons: Sequence[LandPolygon]) -> float:
    """Distance to the nearest coast (0 on land, measured to the shore inside a lagoon); inf without land."""
    scale_x = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat))
    best = math.inf
    for polygon in polygons:
        if _bounds_distance_m(lon, lat, polygon.bounds, scale_x) >= best:
            continue
        if _inside(lon, lat, polygon.exterior) and not any(
            _inside(lon, lat, h) for h in polygon.holes
        ):
            return 0.0
        for ring in (polygon.exterior, *polygon.holes):
            best = min(best, _ring_distance_m(lon, lat, ring, scale_x))
    return best


def distance_to_box_edge_m(lon: float, lat: float, box: BBox) -> float:
    """Distance from a point inside ``box`` to its nearest side; 0 outside it."""
    if not box.contains(lon, lat):
        return 0.0
    scale_x = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat))
    return min(
        (lon - box.min_lon) * scale_x,
        (box.max_lon - lon) * scale_x,
        (lat - box.min_lat) * M_PER_DEG_LAT,
        (box.max_lat - lat) * M_PER_DEG_LAT,
    )


def prefilter(
    detections: Sequence[SarDetection],
    *,
    land: Sequence[LandPolygon],
    infrastructure: InfrastructureMask,
    box: BBox,
    shore_m: float,
    edge_m: float,
    merge_m: float = DUPLICATE_RADIUS_M,
) -> list[Prefiltered]:
    """Each detection's first exclusion reason (duplicate, shore, infrastructure, edge) or None if kept."""
    keeper = merge_duplicates(detections, merge_m)
    results: list[Prefiltered] = []
    for i, d in enumerate(detections):
        shore = distance_to_land_m(d.lon, d.lat, land)
        edge = distance_to_box_edge_m(d.lon, d.lat, box)
        if keeper[i] != i:
            reason = DUPLICATE
        elif shore <= shore_m:
            reason = SHORE
        elif infrastructure.covers(d.lon, d.lat):
            reason = INFRASTRUCTURE
        elif edge <= edge_m:
            reason = EDGE
        else:
            reason = None
        results.append(
            Prefiltered(
                reason=reason,
                merged_into=keeper[i] if keeper[i] != i else None,
                shore_m=shore,
                edge_m=edge,
            )
        )
    return results


def _inside(lon: float, lat: float, ring: np.ndarray) -> bool:
    """Even-odd ray casting; closed or open rings both work."""
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    crosses = (y > lat) != (y2 > lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x + (lat - y) * (x2 - x) / (y2 - y)
    return bool(np.count_nonzero(crosses & (lon < x_at)) % 2)


def _ring_distance_m(lon: float, lat: float, ring: np.ndarray, scale_x: float) -> float:
    """Distance from the point to the nearest edge of a ring, projected about the point."""
    px = (ring[:, 0] - lon) * scale_x
    py = (ring[:, 1] - lat) * M_PER_DEG_LAT
    dx, dy = np.roll(px, -1) - px, np.roll(py, -1) - py
    length2 = dx * dx + dy * dy
    t = np.clip(-(px * dx + py * dy) / np.where(length2 > 0, length2, 1.0), 0.0, 1.0)
    return float(np.min(np.hypot(px + t * dx, py + t * dy)))


def _bounds_distance_m(
    lon: float, lat: float, bounds: tuple[float, float, float, float], scale_x: float
) -> float:
    """Lower bound on the distance to anything inside (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    dx = max(min_lon - lon, 0.0, lon - max_lon) * scale_x
    dy = max(min_lat - lat, 0.0, lat - max_lat) * M_PER_DEG_LAT
    return math.hypot(dx, dy)
