"""Persistent spots: where something is detected pass after pass.

GFW removes such objects from its vessel detections as infrastructure; our
platform mask (OPD) misses jetties, moorings, buoys and unmapped platforms,
and a ship moored for months looks the same (PLAN §17, 2026-10-05/06: about
38% of the model's false alarms against GFW off Iran recur at fixed spots).
A spot is set by a rule fixed before any result: detected within RADIUS_M
on at least MIN_PASSES passes and on at least MIN_SHARE of the passes of
its box in the archive. Spots are built offline (scripts/build_persistence.py)
and read by the API, which marks detections and dark-vessel candidates near
them; nothing is hidden. Pure Python: the API imports it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from core.geo import haversine_m

RADIUS_M = 150.0
MIN_PASSES = 3
MIN_SHARE = 0.3
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQUATOR = 111_320.0


@dataclass(frozen=True)
class Spot:
    """A place where something was detected on ``passes_seen`` of ``passes_imaged`` passes."""

    lon: float
    lat: float
    area: str
    passes_seen: int
    passes_imaged: int
    first: str
    last: str


def find_spots(
    detections: Sequence[tuple[str, str, float, float]],
    passes_per_area: Mapping[str, int],
    *,
    radius_m: float = RADIUS_M,
    min_passes: int = MIN_PASSES,
    min_share: float = MIN_SHARE,
) -> list[Spot]:
    """Persistent spots from (area, scene, lon, lat) detections of archive passes.

    A detection seeds a spot when detections of at least ``min_passes``
    distinct passes (its own included) lie within ``radius_m`` of it, and
    they are at least ``min_share`` of its area's passes. Seeds seen on more
    passes come first; a seed takes every detection within ``radius_m`` (the
    spot is their mean position), so one object detected with some jitter is
    one spot.
    """
    by_area: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for area, scene, lon, lat in detections:
        by_area[area].append((scene, lon, lat))
    spots: list[Spot] = []
    for area in sorted(by_area):
        points = by_area[area]
        imaged = passes_per_area[area]
        near = _neighbours(points, radius_m)
        seen = [{points[j][0] for j in near[i]} for i in range(len(points))]
        seeds = sorted(
            (
                i
                for i in range(len(points))
                if _persistent(len(seen[i]), imaged, min_passes, min_share)
            ),
            key=lambda i: (-len(seen[i]), i),
        )
        taken: set[int] = set()
        for i in seeds:
            if i in taken:
                continue
            members = [j for j in near[i] if j not in taken]
            taken.update(members)
            days = sorted(_day(scene) for scene in seen[i])
            spots.append(
                Spot(
                    lon=sum(points[j][1] for j in members) / len(members),
                    lat=sum(points[j][2] for j in members) / len(members),
                    area=area,
                    passes_seen=len(seen[i]),
                    passes_imaged=imaged,
                    first=days[0],
                    last=days[-1],
                )
            )
    return spots


def spot_near(
    lon: float, lat: float, spots: Sequence[Spot], radius_m: float = RADIUS_M
) -> Spot | None:
    """The nearest spot within ``radius_m``, or None."""
    best, best_m = None, radius_m
    for spot in spots:
        if abs(spot.lat - lat) * _M_PER_DEG_LAT > radius_m:
            continue
        distance = haversine_m(lon, lat, spot.lon, spot.lat)
        if distance <= best_m:
            best, best_m = spot, distance
    return best


def save_spots(path: Path, spots: Sequence[Spot], meta: Mapping[str, object]) -> None:
    """Spots with the rule that made them and what they were built from."""
    rule = {"radius_m": RADIUS_M, "min_passes": MIN_PASSES, "min_share": MIN_SHARE}
    body = {"rule": rule, **meta, "spots": [asdict(s) for s in spots]}
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")


def load_spots(path: Path) -> tuple[dict, list[Spot]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    spots = [Spot(**s) for s in body.pop("spots")]
    return body, spots


def _persistent(passes: int, imaged: int, min_passes: int, min_share: float) -> bool:
    return passes >= min_passes and passes >= min_share * imaged


def _neighbours(points: Sequence[tuple[str, float, float]], radius_m: float) -> list[list[int]]:
    """For each point, the indices of points within ``radius_m`` (itself included)."""
    if not points:
        return []
    widest = max(abs(lat) for _, _, lat in points)
    cell_lat = radius_m / _M_PER_DEG_LAT
    cell_lon = radius_m / (_M_PER_DEG_LON_EQUATOR * math.cos(math.radians(widest)))
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    keys = []
    for i, (_, lon, lat) in enumerate(points):
        key = (math.floor(lat / cell_lat), math.floor(lon / cell_lon))
        grid[key].append(i)
        keys.append(key)
    near = []
    for i, (_, lon, lat) in enumerate(points):
        row, col = keys[i]
        near.append(
            [
                j
                for d_row in (-1, 0, 1)
                for d_col in (-1, 0, 1)
                for j in grid.get((row + d_row, col + d_col), ())
                if haversine_m(lon, lat, points[j][1], points[j][2]) <= radius_m
            ]
        )
    return near


def _day(scene: str) -> str:
    stamp = scene.split("_")[4]
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
