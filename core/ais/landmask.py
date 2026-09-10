"""A conservative land mask for the Persian Gulf.

Purpose and limits
------------------
This exists to catch *gross* falsification - a tanker reporting a position at
an inland airport, or in the middle of the Iranian plateau - which was widely
observed around Hormuz during the 2026 GNSS jamming.

It is deliberately COARSE AND CONSERVATIVE. Each polygon is drawn well inside
the true coastline, so the entire coastal strip, every port, and every
anchorage read as water. The cost is that a position a few kilometres inland
is not flagged; the benefit is that a vessel alongside in Jebel Ali is never
called a spoof. For this check a false negative is cheap and a false positive
is not.

Phase 2 downloads GSHHG for the SAR land mask. When it does, swap
:class:`PolygonLandMask` for a GSHHG-backed implementation behind the same
interface and this file becomes a fallback.
"""

from __future__ import annotations

from typing import Protocol

Polygon = list[tuple[float, float]]


class LandMask(Protocol):
    """Anything that can answer 'is this position confidently on land?'."""

    def is_land(self, lon: float, lat: float) -> bool: ...


# Each polygon sits INSIDE the real landmass, offset roughly 0.5-1.0 degrees
# from the coast. Coordinates are (lon, lat).
_IRAN_INTERIOR: Polygon = [
    (47.6, 31.4),
    (49.2, 31.6),
    (50.4, 31.2),
    (51.4, 30.0),
    (52.4, 29.0),
    (53.6, 28.4),
    (55.0, 28.0),
    (56.6, 28.4),
    (58.2, 28.0),
    (58.6, 33.5),
    (47.2, 33.5),
]

_ARABIAN_INTERIOR: Polygon = [
    (47.4, 28.6),
    (48.6, 27.6),
    (49.4, 26.4),
    (50.0, 25.2),
    (50.4, 24.2),
    (52.6, 23.2),
    (55.0, 23.2),
    (56.2, 24.2),
    (56.4, 22.5),
    (46.0, 22.5),
    (46.0, 28.6),
]

# Qatar is a narrow peninsula, so only its centre is safely inland.
_QATAR_INTERIOR: Polygon = [
    (51.0, 25.0),
    (51.3, 25.0),
    (51.3, 25.7),
    (51.0, 25.7),
]

# The Omani interior. Held east of about 56.6 deg E because Fujairah and the
# UAE east coast sit near 56.35 deg E with deep-water anchorage immediately
# offshore - a polygon reaching that far west would flag ships at anchor.
_OMAN_INTERIOR: Polygon = [
    (56.7, 24.4),
    (57.6, 23.2),
    (58.8, 23.4),
    (58.8, 25.6),
    (57.2, 25.6),
    (56.9, 24.8),
]

# The Musandam peninsula, kept small: the strait's traffic passes just north
# and east of it.
_MUSANDAM: Polygon = [
    (56.20, 25.95),
    (56.40, 25.95),
    (56.40, 26.25),
    (56.20, 26.25),
]

_LAND_POLYGONS: list[Polygon] = [
    _IRAN_INTERIOR,
    _ARABIAN_INTERIOR,
    _QATAR_INTERIOR,
    _OMAN_INTERIOR,
    _MUSANDAM,
]


def point_in_polygon(lon: float, lat: float, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test.

    Implemented directly rather than via shapely: the polygons are tiny and
    this keeps the ingest path free of a geometry dependency.
    """
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        # Does the edge straddle the horizontal ray at `lat`?
        if (y1 > lat) != (y2 > lat):
            x_at_lat = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_at_lat:
                inside = not inside
    return inside


class PolygonLandMask:
    """Land mask backed by the conservative interior polygons above."""

    def __init__(self, polygons: list[Polygon] | None = None) -> None:
        self._polygons = polygons if polygons is not None else _LAND_POLYGONS

    def is_land(self, lon: float, lat: float) -> bool:
        return any(point_in_polygon(lon, lat, poly) for poly in self._polygons)


class NoLandMask:
    """A mask that never reports land. Used when the check is disabled."""

    def is_land(self, lon: float, lat: float) -> bool:  # noqa: ARG002
        return False


DEFAULT_LAND_MASK: LandMask = PolygonLandMask()
