"""Pixel <-> geographic coordinates for SAR scenes.

Sentinel-1 GRD scenes are not map-projected
-------------------------------------------
The Cloud-Optimized GRD products CDSE serves are still delivered in radar
geometry. Opened with GDAL they have no CRS and an identity geotransform;
location is carried instead by a grid of ground control points (210 on an IW
scene) in EPSG:4326. Code that treats such a raster as affine places every
pixel at longitude = column and latitude = row - which is to say, nowhere.
This module is the one place that knows how to convert.

Two further facts, measured on a real scene (S1D IW GRDH, 2026-09-10, a
descending pass over the Gulf of Oman):

* **The image is mirrored.** Columns run from east to west, so a heading
  computed from pixel axes comes out reflected. Bearings are therefore always
  computed geographically, between two transformed points.
* **The interpolation method decides the accuracy.** GDAL's default
  polynomial fit misplaced the control points themselves by up to 1.2 km. A
  thin-plate spline reproduces them exactly and interpolates between them
  with a leave-one-out error of 26 m mean and 55 m p90 (233 m worst case, at
  the scene edge where a removed point forces extrapolation). AIS matching
  gates at a few hundred metres, so the spline is not optional.

Pixel conventions
-----------------
GDAL coordinates are corner-based: pixel index ``i`` spans ``[i, i + 1)``.
:meth:`SceneGeoref.lonlat` takes pixel *indices* and treats them as pixel
centres, which is what a component centroid computed from integer indices
is. :meth:`SceneGeoref.raster_coords` returns continuous corner-based
coordinates; ``floor`` of one is the index of the pixel containing the point.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.db.spatial import BBox
from core.geo import bearing_deg, haversine_m
from core.sar.windows import PixelWindow, clamp_window

# Points sampled along each bbox edge. Under a non-linear mapping a box's
# pixel extent is not determined by its four corners alone.
DENSIFY_POINTS = 21

# Pixel distance over which local scale and direction are measured: long
# enough that interpolation noise is negligible, short enough to stay local.
PROBE_PIXELS = 50.0

WGS84_EPSG = 4326

ArrayLike = Sequence[float] | np.ndarray | float


@dataclass(frozen=True)
class SceneGeoref:
    """How one raster's pixels map to the ground.

    Exactly one of ``affine`` (a geotransform, geographic unless ``crs`` says
    otherwise) or ``gcps`` (radar geometry located by ground control points)
    is set. ``width`` and ``height`` are only needed for window clipping.
    """

    width: int = 0
    height: int = 0
    affine: Any = None
    crs: Any = None
    gcps: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(f"raster size cannot be negative: {self.width}x{self.height}")
        if (self.affine is not None) == bool(self.gcps):
            raise ValueError("provide exactly one of an affine transform or ground control points")

    @classmethod
    def from_dataset(cls, dataset: Any) -> SceneGeoref:
        """Build from an open rasterio dataset, preferring its control points."""
        gcps, gcp_crs = dataset.gcps
        if gcps:
            if gcp_crs is not None and gcp_crs.to_epsg() != WGS84_EPSG:
                raise ValueError(f"ground control points must be EPSG:4326, got {gcp_crs}")
            return cls(width=dataset.width, height=dataset.height, gcps=tuple(gcps))

        if dataset.crs is None or dataset.transform.is_identity:
            raise ValueError(
                "raster has neither a geotransform nor ground control points, "
                "so its pixels cannot be located"
            )
        return cls(
            width=dataset.width,
            height=dataset.height,
            affine=dataset.transform,
            crs=dataset.crs,
        )

    @classmethod
    def from_affine(
        cls, transform: Any, width: int = 0, height: int = 0, crs: Any = None
    ) -> SceneGeoref:
        """A raster whose transform maps to lon/lat, unless ``crs`` is projected."""
        return cls(width=width, height=height, affine=transform, crs=crs)

    @property
    def uses_gcps(self) -> bool:
        return bool(self.gcps)

    @property
    def _projected(self) -> bool:
        return self.crs is not None and not self.crs.is_geographic

    @contextmanager
    def _transformer(self) -> Iterator[Any]:
        from rasterio.transform import AffineTransformer, GCPTransformer

        if not self.gcps:
            yield AffineTransformer(self.affine)
            return
        transformer = GCPTransformer(list(self.gcps), tps=True)
        try:
            yield transformer
        finally:
            transformer.close()

    def lonlat(self, rows: ArrayLike, cols: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Longitude and latitude of pixel centres at (possibly fractional) indices."""
        row_arr = np.atleast_1d(np.asarray(rows, dtype=np.float64))
        col_arr = np.atleast_1d(np.asarray(cols, dtype=np.float64))
        with self._transformer() as transformer:
            xs, ys = transformer.xy(row_arr, col_arr, offset="center")
        lons = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        lats = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        if self._projected:
            lons, lats = _reproject(self.crs, "EPSG:4326", lons, lats)
        return lons, lats

    def raster_coords(self, lons: ArrayLike, lats: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Continuous, corner-based (row, col) raster coordinates of points."""
        lon_arr = np.atleast_1d(np.asarray(lons, dtype=np.float64))
        lat_arr = np.atleast_1d(np.asarray(lats, dtype=np.float64))
        if self._projected:
            lon_arr, lat_arr = _reproject("EPSG:4326", self.crs, lon_arr, lat_arr)
        with self._transformer() as transformer:
            # rasterio floors to integers unless given ``op``; ``float`` keeps
            # the fractional part.
            rows, cols = transformer.rowcol(lon_arr, lat_arr, op=float)
        return (
            np.atleast_1d(np.asarray(rows, dtype=np.float64)),
            np.atleast_1d(np.asarray(cols, dtype=np.float64)),
        )

    def window_for_bbox(self, bbox: BBox, *, densify: int = DENSIFY_POINTS) -> PixelWindow:
        """The pixel window covering a geographic box, clipped to the raster.

        An empty window means the box does not reach the raster. Because the
        box is sampled along its edges and the result is axis-aligned in
        pixel space, a rotated scene can yield a window slightly larger than
        strictly necessary - never smaller, which is the direction that matters.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("raster dimensions are needed to clip a window")
        lons, lats = _densified_ring(bbox, densify)
        rows, cols = self.raster_coords(lons, lats)
        finite = np.isfinite(rows) & np.isfinite(cols)
        if not finite.any():
            return PixelWindow(0, 0, 0, 0)

        row0 = math.floor(float(rows[finite].min()))
        row1 = math.ceil(float(rows[finite].max()))
        col0 = math.floor(float(cols[finite].min()))
        col1 = math.ceil(float(cols[finite].max()))
        window = PixelWindow(col0, row0, col1 - col0, row1 - row0)
        return clamp_window(window, self.width, self.height)

    def metres_per_pixel(self, row: float, col: float) -> tuple[float, float]:
        """Ground distance of one pixel step along columns, then along rows."""
        lons, lats = self.lonlat(
            [row, row, row + PROBE_PIXELS],
            [col, col + PROBE_PIXELS, col],
        )
        along_col = haversine_m(lons[0], lats[0], lons[1], lats[1]) / PROBE_PIXELS
        along_row = haversine_m(lons[0], lats[0], lons[2], lats[2]) / PROBE_PIXELS
        return along_col, along_row

    def bearing(self, row: float, col: float, drow: float, dcol: float) -> float:
        """Compass bearing, in degrees, of a pixel-space direction at a location."""
        return float(self.bearings([row], [col], [drow], [dcol])[0])

    def bearings(
        self, rows: ArrayLike, cols: ArrayLike, drows: ArrayLike, dcols: ArrayLike
    ) -> np.ndarray:
        """Vectorised :meth:`bearing`, sharing one transformer across all points."""
        row_arr = np.atleast_1d(np.asarray(rows, dtype=np.float64))
        col_arr = np.atleast_1d(np.asarray(cols, dtype=np.float64))
        drow_arr = np.atleast_1d(np.asarray(drows, dtype=np.float64))
        dcol_arr = np.atleast_1d(np.asarray(dcols, dtype=np.float64))
        norms = np.hypot(drow_arr, dcol_arr)
        if np.any(norms == 0.0):
            raise ValueError("a direction must be non-zero")

        scale = PROBE_PIXELS / norms
        lons, lats = self.lonlat(
            np.concatenate([row_arr, row_arr + drow_arr * scale]),
            np.concatenate([col_arr, col_arr + dcol_arr * scale]),
        )
        count = row_arr.size
        return np.array(
            [bearing_deg(lons[i], lats[i], lons[count + i], lats[count + i]) for i in range(count)]
        )


def _densified_ring(bbox: BBox, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Points along a bbox's boundary, anticlockwise from the south-west corner."""
    steps = np.linspace(0.0, 1.0, max(points, 2))
    edge_lons = bbox.min_lon + steps * (bbox.max_lon - bbox.min_lon)
    edge_lats = bbox.min_lat + steps * (bbox.max_lat - bbox.min_lat)
    lons = np.concatenate(
        [
            edge_lons,
            np.full_like(steps, bbox.max_lon),
            edge_lons[::-1],
            np.full_like(steps, bbox.min_lon),
        ]
    )
    lats = np.concatenate(
        [
            np.full_like(steps, bbox.min_lat),
            edge_lats,
            np.full_like(steps, bbox.max_lat),
            edge_lats[::-1],
        ]
    )
    return lons, lats


def _reproject(src: Any, dst: Any, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from rasterio.warp import transform

    out_x, out_y = transform(src, dst, xs.tolist(), ys.tolist())
    return np.asarray(out_x, dtype=np.float64), np.asarray(out_y, dtype=np.float64)
