"""The five study boxes, the detectors tested in them, and the area a run searched.

The product model was trained, validated and tested on these five boxes off
the UAE (2024-25, Sentinel-1A) and checked there on Sentinel-1C/1D (2026-09),
against Global Fishing Watch and next to the CFAR baseline. Anywhere else -
the Iranian coast, the northern Gulf - and with any other detector
configuration, accuracy has not been measured, and the console says so. Also
the box every detection run and comparison uses (a square of ``size_km`` on
the ground around a centre) and the part of it a pass actually imaged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.db.spatial import BBox

KM_PER_DEG_LAT = 110.54
KM_PER_DEG_LON_EQUATOR = 111.32
# Centre longitude, centre latitude, side in km.
STUDY_AREAS: dict[str, tuple[float, float, float]] = {
    "fujairah": (56.53, 25.22, 20.0),
    "khor_fakkan": (56.52, 25.41, 20.0),
    "sharjah_dubai": (55.33, 25.47, 20.0),
    "ras_al_khaimah": (55.88, 26.02, 20.0),
    "hormuz_lanes": (56.55, 26.50, 40.0),
}
# Seven 20 km boxes at anchorages and terminals on the Iranian side, where the
# console runs the product model (data/logs/run_detect_iran.sh). Not study
# boxes: the model was never trained, chosen or tested here, so they are a
# held-out region for scripts/compare_cfar_model.py (--pairs, list_pairs.py).
IRAN_AREAS: dict[str, tuple[float, float, float]] = {
    "bandar_abbas": (56.25, 27.03, 20.0),
    "larak_hormuz": (56.45, 26.85, 20.0),
    "kish": (53.98, 26.53, 20.0),
    "assaluyeh": (52.55, 27.40, 20.0),
    "bushehr": (50.72, 28.95, 20.0),
    "kharg": (50.30, 29.25, 20.0),
    "jask": (57.80, 25.60, 20.0),
}
# Detector, version and polarisation measured against GFW in the study boxes
# (PLAN §17): run 2, the product model, on VV+VH, and the CFAR baseline on VH
# at Pfa 1e-6. Run 1 (kaggle-v1) was measured too, lost to CFAR and was
# superseded, so its searches do not count as tested.
# The product model: run 2, epoch 5 (models/point_detector_kaggle-v2.onnx).
PRODUCT_DETECTOR = "pointdet"
PRODUCT_VERSION = "kaggle-v2"
TESTED_DETECTORS = frozenset(
    {("pointdet", "kaggle-v2", "DV"), ("cfar", "L4.4-g24-t40-p1e-06-h0.25", "VH")}
)
# Clipped vertices can stray past a box edge by rounding.
EDGE_TOLERANCE_DEG = 1e-9

Point = tuple[float, float]


def box_around(lon: float, lat: float, size_km: float) -> BBox:
    half = size_km / 2.0
    dlat = half / KM_PER_DEG_LAT
    dlon = half / (KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)))
    return BBox(lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def study_area_of(lon: float, lat: float) -> str | None:
    """The study box a point lies in, or None where the detector was never measured."""
    for name, (centre_lon, centre_lat, size_km) in STUDY_AREAS.items():
        if box_around(centre_lon, centre_lat, size_km).contains(lon, lat):
            return name
    return None


def is_tested(detector: str, version: str | None, polarisation: str | None) -> bool:
    """Whether this detector configuration was measured against GFW (in the study boxes)."""
    return (detector, version, (polarisation or "").upper()) in TESTED_DETECTORS


def _at_lon(a: Point, b: Point, lon: float) -> Point:
    t = (lon - a[0]) / (b[0] - a[0])
    return (lon, a[1] + t * (b[1] - a[1]))


def _at_lat(a: Point, b: Point, lat: float) -> Point:
    t = (lat - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), lat)


def clip_to_box(ring: Sequence[Point], bbox: BBox) -> list[Point]:
    """The part of a polygon inside a box, as an open ring; [] if they do not overlap.

    Sutherland-Hodgman against the box's four edges, exact for a simple
    polygon against the convex box. Cuts a requested detection box down to
    the part a pass imaged: a box can reach beyond the swath edge.
    """
    points = [(float(lon), float(lat)) for lon, lat in ring]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    edges = (
        (lambda p: p[0] >= bbox.min_lon, lambda a, b: _at_lon(a, b, bbox.min_lon)),
        (lambda p: p[0] <= bbox.max_lon, lambda a, b: _at_lon(a, b, bbox.max_lon)),
        (lambda p: p[1] >= bbox.min_lat, lambda a, b: _at_lat(a, b, bbox.min_lat)),
        (lambda p: p[1] <= bbox.max_lat, lambda a, b: _at_lat(a, b, bbox.max_lat)),
    )
    for inside, crossing in edges:
        kept: list[Point] = []
        for i, current in enumerate(points):
            previous = points[i - 1]
            if inside(current):
                if not inside(previous):
                    kept.append(crossing(previous, current))
                kept.append(current)
            elif inside(previous):
                kept.append(crossing(previous, current))
        points = kept
        if not points:
            break
    return points if len(points) >= 3 else []


def ring_area(ring: Sequence[Point]) -> float:
    """Planar area of a ring in square degrees (shoelace; open or closed rings alike).

    Only ratios over one small box are taken, where the cos(latitude) factor cancels.
    """
    points = list(ring)
    twice = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True)
    )
    return abs(twice) / 2.0


def inside_study_box(ring: Sequence[Point]) -> str | None:
    """The study box wholly containing an area, or None (also for an empty area)."""
    if not ring:
        return None
    for name, (lon, lat, size_km) in STUDY_AREAS.items():
        box = box_around(lon, lat, size_km)
        if all(
            box.min_lon - EDGE_TOLERANCE_DEG <= x <= box.max_lon + EDGE_TOLERANCE_DEG
            and box.min_lat - EDGE_TOLERANCE_DEG <= y <= box.max_lat + EDGE_TOLERANCE_DEG
            for x, y in ring
        ):
            return name
    return None
