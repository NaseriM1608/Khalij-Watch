"""Coastline land mask for SAR windows.

Source: OpenStreetMap land polygons (osmdata.openstreetmap.de), derived from
the OpenStreetMap coastline. (c) OpenStreetMap contributors, ODbL 1.0.

Why OpenStreetMap rather than GSHHG
-----------------------------------
PLAN.md named GSHHG. Its full-resolution shoreline was last released in 2017
and is built from 1990s sources, and few coasts have changed as much since as
this one: Palm Jumeirah, the Dubai and Doha reclamations, new ports. Every
such structure missing from the mask is a bright, permanent false alarm.
OpenStreetMap is traced from recent imagery and rebuilt daily.

The buffer
----------
Pixels within ``buffer_m`` of land are masked too; the buffer is a square
dilation, so along diagonals it reaches about 1.4 times as far. It absorbs
coastline error, georeferencing error (the thin-plate spline's leave-one-out
error is 26 m mean and 55 m p90) and shoreline structures such as piers and
breakwaters, which OpenStreetMap draws as lines rather than land. The cost is
that vessels alongside a quay are not detected.

Reference file
--------------
The polygons around the area of interest are extracted once, by
``scripts/fetch_coastline.py``, into a single ``.npz`` that needs only numpy
to load: ``coords`` (N x 2 lon, lat), ``ring_starts`` (N rings + 1 offsets),
``ring_polygon`` (owning polygon of each ring), ``ring_is_hole``,
``polygon_bbox`` (P x 4: min lon, min lat, max lon, max lat) and ``meta``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.db.spatial import BBox
from core.sar.georef import SceneGeoref
from core.sar.morphology import dilate

DEFAULT_PATH = Path("data") / "reference" / "osm_land_polygons_gulf.npz"
ATTRIBUTION = "Coastline (c) OpenStreetMap contributors, ODbL 1.0, via osmdata.openstreetmap.de"

# Provisional. On a real Fujairah coast window (2026-09-10) the candidates in
# a 20 km box fell from 87 unbuffered to 46 at 150 m and 32 at 500 m; 500 m
# also blinds port approaches. See PLAN.md section 17.
DEFAULT_BUFFER_M = 150.0

# Land is gathered this many pixels beyond the buffered window, so the edges
# that clipping cuts into polygons never fall inside the rasterised canvas.
CLIP_MARGIN_PX = 16

# Points sampled along each window edge to find its geographic reach.
EDGE_SAMPLES = 21


@dataclass(frozen=True, eq=False)
class LandPolygon:
    """One land polygon: an exterior ring and any holes, each an (n, 2) lon/lat array."""

    exterior: np.ndarray
    holes: tuple[np.ndarray, ...] = ()
    bounds: tuple[float, float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        exterior = np.asarray(self.exterior, dtype=np.float64)
        holes = tuple(np.asarray(hole, dtype=np.float64) for hole in self.holes)
        for ring in (exterior, *holes):
            if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3:
                raise ValueError("every ring needs at least three (lon, lat) points")
        object.__setattr__(self, "exterior", exterior)
        object.__setattr__(self, "holes", holes)
        object.__setattr__(
            self,
            "bounds",
            (
                float(exterior[:, 0].min()),
                float(exterior[:, 1].min()),
                float(exterior[:, 0].max()),
                float(exterior[:, 1].max()),
            ),
        )


def clip_ring(ring: np.ndarray, bbox: BBox) -> np.ndarray:
    """Clip a ring to an axis-aligned box (Sutherland-Hodgman).

    Returns the clipped vertices, possibly none. A concave ring can come back
    with zero-width edges along the box boundary; they enclose no area, and
    :class:`CoastlineMask` keeps the boundary outside the pixels it fills.
    """
    points = np.asarray(ring, dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 2)
    inside = (
        (points[:, 0] >= bbox.min_lon)
        & (points[:, 0] <= bbox.max_lon)
        & (points[:, 1] >= bbox.min_lat)
        & (points[:, 1] <= bbox.max_lat)
    )
    if inside.all():
        return points
    for axis, limit, keep_above in (
        (0, bbox.min_lon, True),
        (0, bbox.max_lon, False),
        (1, bbox.min_lat, True),
        (1, bbox.max_lat, False),
    ):
        if len(points) == 0:
            break
        points = _clip_half_plane(points, axis, limit, keep_above)
    return points.reshape(-1, 2)


def _clip_half_plane(points: np.ndarray, axis: int, limit: float, keep_above: bool) -> np.ndarray:
    """One Sutherland-Hodgman pass, vectorised over the ring's edges."""
    previous = np.roll(points, 1, axis=0)
    inside = points[:, axis] >= limit if keep_above else points[:, axis] <= limit
    crossing = inside != np.roll(inside, 1)
    delta = points[:, axis] - previous[:, axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(crossing, (limit - previous[:, axis]) / delta, 0.0)
    intersections = previous + t[:, None] * (points - previous)
    # Each edge emits its crossing point, if any, and then its end point if kept.
    candidates = np.stack([intersections, points], axis=1)
    return candidates[np.stack([crossing, inside], axis=1)]


class CoastlineMask:
    """Rasterises land, plus a seaward buffer, into a per-window exclusion mask."""

    def __init__(
        self, polygons: Sequence[LandPolygon], *, buffer_m: float = DEFAULT_BUFFER_M
    ) -> None:
        if buffer_m < 0:
            raise ValueError(f"buffer must be non-negative, got {buffer_m}")
        self._polygons = tuple(polygons)
        self._bounds = np.array([p.bounds for p in self._polygons], dtype=np.float64).reshape(-1, 4)
        self._buffer_m = buffer_m

    def __len__(self) -> int:
        return len(self._polygons)

    def rasterise(
        self,
        scene: SceneGeoref,
        width: int,
        height: int,
        *,
        window_offset: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Boolean mask of land and buffer for a window of ``scene``.

        ``window_offset`` is the (row, col) origin of the window in the scene.
        Land outside the window still counts: its buffer can reach inside.
        """
        from affine import Affine
        from rasterio import features

        empty = np.zeros((max(height, 0), max(width, 0)), dtype=bool)
        if width <= 0 or height <= 0 or not self._polygons:
            return empty

        row_off, col_off = window_offset
        pad = self._buffer_pixels(scene, row_off + height / 2.0, col_off + width / 2.0)
        margin = pad + CLIP_MARGIN_PX
        reach = _reach(
            scene, row_off - margin, col_off - margin, height + 2 * margin, width + 2 * margin
        )

        shapes = _pixel_shapes(scene, self._clipped_to(reach), row_off - pad, col_off - pad)
        if not shapes:
            return empty
        canvas = features.rasterize(
            ((shape, 1) for shape in shapes),
            out_shape=(height + 2 * pad, width + 2 * pad),
            transform=Affine.identity(),
            fill=0,
            dtype="uint8",
        ).astype(bool)
        if pad:
            canvas = dilate(canvas, pad)
        return canvas[pad : pad + height, pad : pad + width]

    def _buffer_pixels(self, scene: SceneGeoref, row: float, col: float) -> int:
        if self._buffer_m == 0:
            return 0
        # The finer of the two spacings, so the buffer is never narrower than asked.
        spacing = min(scene.metres_per_pixel(row, col))
        return math.ceil(self._buffer_m / max(spacing, 1e-6))

    def _clipped_to(self, reach: BBox) -> list[list[np.ndarray]]:
        """Rings of every polygon touching ``reach``, clipped to it; exterior first."""
        b = self._bounds
        touching = np.flatnonzero(
            (b[:, 2] >= reach.min_lon)
            & (b[:, 0] <= reach.max_lon)
            & (b[:, 3] >= reach.min_lat)
            & (b[:, 1] <= reach.max_lat)
        )
        kept: list[list[np.ndarray]] = []
        for index in touching:
            polygon = self._polygons[index]
            exterior = clip_ring(polygon.exterior, reach)
            if len(exterior) < 3:
                continue
            holes = [
                ring for ring in (clip_ring(h, reach) for h in polygon.holes) if len(ring) >= 3
            ]
            kept.append([exterior, *holes])
        return kept


def _reach(scene: SceneGeoref, row0: float, col0: float, height: float, width: float) -> BBox:
    """Geographic box around a pixel rectangle, sampled along its edges."""
    along_rows = np.linspace(row0, row0 + height, EDGE_SAMPLES)
    along_cols = np.linspace(col0, col0 + width, EDGE_SAMPLES)
    rows = np.concatenate(
        [along_rows, along_rows, np.full(EDGE_SAMPLES, row0), np.full(EDGE_SAMPLES, row0 + height)]
    )
    cols = np.concatenate(
        [np.full(EDGE_SAMPLES, col0), np.full(EDGE_SAMPLES, col0 + width), along_cols, along_cols]
    )
    lons, lats = scene.lonlat(rows, cols)
    return BBox(float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()))


def _pixel_shapes(
    scene: SceneGeoref, polygons: list[list[np.ndarray]], row_origin: float, col_origin: float
) -> list[dict]:
    """GeoJSON polygons in canvas pixel coordinates (x = column, y = row).

    Every vertex goes through one transformer call: building a thin-plate
    spline for each ring separately would dominate the cost.
    """
    rings = [ring for polygon in polygons for ring in polygon]
    if not rings:
        return []
    offsets = np.concatenate([[0], np.cumsum([len(ring) for ring in rings])])
    lonlat = np.concatenate(rings)
    rows, cols = scene.raster_coords(lonlat[:, 0], lonlat[:, 1])
    xs, ys = cols - col_origin, rows - row_origin

    shapes: list[dict] = []
    ring_index = 0
    for polygon in polygons:
        coordinates = []
        for _ in polygon:
            start, stop = offsets[ring_index], offsets[ring_index + 1]
            ring_index += 1
            ring = np.column_stack([xs[start:stop], ys[start:stop]])
            if not np.isfinite(ring).all():
                continue
            if not np.array_equal(ring[0], ring[-1]):
                ring = np.vstack([ring, ring[:1]])
            coordinates.append(ring.tolist())
        # A polygon whose exterior could not be placed is dropped with its holes.
        if coordinates and len(coordinates) == len(polygon):
            shapes.append({"type": "Polygon", "coordinates": coordinates})
    return shapes


def save_polygons(path: Path, polygons: Sequence[LandPolygon], *, source: str) -> None:
    """Write polygons to the reference ``.npz`` format described above."""
    rings: list[np.ndarray] = []
    owners: list[int] = []
    is_hole: list[bool] = []
    for index, polygon in enumerate(polygons):
        for ring, hole in ((polygon.exterior, False), *((h, True) for h in polygon.holes)):
            rings.append(ring)
            owners.append(index)
            is_hole.append(hole)
    counts = np.array([len(ring) for ring in rings], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords=np.concatenate(rings) if rings else np.zeros((0, 2), dtype=np.float64),
        ring_starts=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        ring_polygon=np.array(owners, dtype=np.int32),
        ring_is_hole=np.array(is_hole, dtype=bool),
        polygon_bbox=np.array([p.bounds for p in polygons], dtype=np.float64).reshape(-1, 4),
        meta=np.array([ATTRIBUTION, source]),
    )


def load_land(path: Path | None = None, *, bbox: BBox | None = None) -> list[LandPolygon]:
    """Load land polygons from the reference file, optionally only those touching ``bbox``."""
    path = path or DEFAULT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Build it with: uv run --with pyshp python scripts/fetch_coastline.py"
        )
    with np.load(path, allow_pickle=False) as data:
        coords = data["coords"]
        starts = data["ring_starts"]
        owners = data["ring_polygon"]
        holes = data["ring_is_hole"]
        bounds = data["polygon_bbox"]

    if bbox is None:
        wanted = np.ones(len(bounds), dtype=bool)
    else:
        wanted = (
            (bounds[:, 2] >= bbox.min_lon)
            & (bounds[:, 0] <= bbox.max_lon)
            & (bounds[:, 3] >= bbox.min_lat)
            & (bounds[:, 1] <= bbox.max_lat)
        )

    exteriors: dict[int, np.ndarray] = {}
    hole_rings: dict[int, list[np.ndarray]] = {}
    for ring_index in np.flatnonzero(wanted[owners]):
        owner = int(owners[ring_index])
        ring = coords[starts[ring_index] : starts[ring_index + 1]]
        if holes[ring_index]:
            hole_rings.setdefault(owner, []).append(ring)
        else:
            exteriors[owner] = ring
    return [
        LandPolygon(exteriors[owner], holes=tuple(hole_rings.get(owner, ())))
        for owner in sorted(exteriors)
    ]
