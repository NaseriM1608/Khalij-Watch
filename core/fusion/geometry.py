"""Local planar distances for cell-scale fusion geometry.

Uses the local equirectangular projection of ``core.geo.linear_deviation_m``
(111,320 m per degree of longitude at the equator, 110,540 m per degree of
latitude). Over the few kilometres of a fusion gate its error against the
ellipsoid is far below the 1.1 km GFW cell the positions are quantised to.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from core.db.spatial import BBox

M_PER_DEG_LON_EQUATOR = 111_320.0
M_PER_DEG_LAT = 110_540.0


def buffer_box(box: BBox, metres: float) -> BBox:
    """``box`` grown by ``metres`` on every side (longitude scaled at its mid-latitude)."""
    if metres < 0:
        raise ValueError(f"buffer {metres} m must be non-negative")
    d_lat = metres / M_PER_DEG_LAT
    d_lon = metres / (
        M_PER_DEG_LON_EQUATOR * math.cos(math.radians((box.min_lat + box.max_lat) / 2))
    )
    return BBox(box.min_lon - d_lon, box.min_lat - d_lat, box.max_lon + d_lon, box.max_lat + d_lat)


def local_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Planar distance between two nearby points, projected about their mean latitude."""
    scale_x = M_PER_DEG_LON_EQUATOR * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lon2 - lon1) * scale_x, (lat2 - lat1) * M_PER_DEG_LAT)


def distance_to_polyline_m(
    lon: float, lat: float, vertices: Sequence[tuple[float, float]]
) -> float:
    """Distance from a point to the nearest point of a (lon, lat) polyline, segments clamped at their ends."""
    if not vertices:
        raise ValueError("a polyline needs at least one vertex")
    scale_x = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat))
    points = [((v_lon - lon) * scale_x, (v_lat - lat) * M_PER_DEG_LAT) for v_lon, v_lat in vertices]
    if len(points) == 1:
        return math.hypot(*points[0])
    return min(_origin_to_segment(a, b) for a, b in pairwise(points))


def _origin_to_segment(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance from (0, 0) to the segment a-b in projected metres."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / length2))
    return math.hypot(a[0] + t * dx, a[1] + t * dy)
