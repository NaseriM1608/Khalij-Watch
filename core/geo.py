"""Geodesic helpers on the WGS84 sphere.

Great-circle distance is accurate to roughly 0.5% against the true ellipsoid,
which is far below the AIS position noise these are used on. Phase 4's SAR
fusion reuses them for match gating.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8
METRES_PER_NAUTICAL_MILE = 1852.0
SECONDS_PER_HOUR = 3600.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two points, in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial bearing from point 1 to point 2, in degrees clockwise from north."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return math.degrees(math.atan2(x, y)) % 360.0


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180].

    Naive subtraction makes 359 deg and 1 deg look 358 apart instead of 2,
    which would flag every vessel crossing north as making an impossible turn.
    """
    return abs((a - b + 180.0) % 360.0 - 180.0)


def speed_kn(distance_m: float, seconds: float) -> float:
    """Average speed implied by covering a distance in a time, in knots."""
    if seconds <= 0:
        return math.inf if distance_m > 0 else 0.0
    metres_per_hour = distance_m / seconds * SECONDS_PER_HOUR
    return metres_per_hour / METRES_PER_NAUTICAL_MILE


def cross_track_distance_m(
    lon: float,
    lat: float,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
) -> float:
    """Perpendicular distance from a point to the great circle through two others.

    Used to detect suspiciously straight tracks: a real vessel wanders, while a
    fabricated one often interpolates a perfect line.
    """
    d13 = haversine_m(start_lon, start_lat, lon, lat) / EARTH_RADIUS_M
    if d13 == 0:
        return 0.0
    theta13 = math.radians(bearing_deg(start_lon, start_lat, lon, lat))
    theta12 = math.radians(bearing_deg(start_lon, start_lat, end_lon, end_lat))
    return abs(math.asin(math.sin(d13) * math.sin(theta13 - theta12)) * EARTH_RADIUS_M)


def linear_deviation_m(
    lon: float,
    lat: float,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
) -> float:
    """Perpendicular distance from a point to the straight lon/lat line.

    Uses a local equirectangular projection about the start point. This is
    deliberately NOT the great-circle test: a naive spoofer interpolates
    linearly in coordinate space, and a constant-latitude track - a very
    common fabrication - is not a great circle at all, so a great-circle
    measure would miss it entirely.
    """
    scale_x = math.cos(math.radians(start_lat)) * 111_320.0
    scale_y = 110_540.0
    px = (lon - start_lon) * scale_x
    py = (lat - start_lat) * scale_y
    ex = (end_lon - start_lon) * scale_x
    ey = (end_lat - start_lat) * scale_y
    segment = math.hypot(ex, ey)
    if segment == 0.0:
        return math.hypot(px, py)
    # Magnitude of the 2-D cross product over the segment length.
    return abs(px * ey - py * ex) / segment
