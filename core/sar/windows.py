"""Windowed reading of Sentinel-1 COG products.

The governing constraint
------------------------
A Sentinel-1 IW GRD scene is roughly 25,000 x 17,000 pixels. As float32 that is
~1.7 GB *per polarisation*, on a host with 10 GB of RAM shared with a browser
and a database. Reading a whole scene is therefore never acceptable, and the
guard here refuses it rather than letting the machine swap to death.

Because CDSE publishes a Cloud-Optimized GeoTIFF variant, an AOI window can be
fetched with HTTP range requests over only the tiles that intersect it - no
full download, no local preprocessing chain, and no SNAP install.

Note what is deliberately absent: **no speckle filtering**. Lee, Refined Lee
and Frost all smear or erase 2-5 pixel targets, which is exactly the signal a
ship gives. The pixels are handed on close to raw and the model handles speckle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.db.spatial import BBox

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    import numpy as np

# Refuse any single allocation larger than this. Sized so that several windows
# can be held at once without approaching the host's 10 GB.
MAX_WINDOW_BYTES = 512 * 1024 * 1024

# Sentinel-1 IW GRDH ground sample distance.
GRDH_PIXEL_SPACING_M = 10.0

# Degrees per metre, near enough for window sizing at Gulf latitudes.
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQUATOR = 111_320.0


class WindowTooLargeError(MemoryError):
    """Raised instead of attempting an allocation that would exhaust RAM."""


@dataclass(frozen=True)
class PixelWindow:
    """A rectangular region in pixel space."""

    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def nbytes(self, dtype_size: int = 4, bands: int = 1) -> int:
        return self.pixels * dtype_size * bands

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0


def estimate_bytes(bbox: BBox, *, dtype_size: int = 4, bands: int = 2) -> int:
    """Approximate memory needed to hold an AOI at Sentinel-1 resolution."""
    mid_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    metres_per_deg_lon = _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(mid_lat))
    width_m = (bbox.max_lon - bbox.min_lon) * metres_per_deg_lon
    height_m = (bbox.max_lat - bbox.min_lat) * _M_PER_DEG_LAT
    cols = max(1, int(width_m / GRDH_PIXEL_SPACING_M))
    rows = max(1, int(height_m / GRDH_PIXEL_SPACING_M))
    return cols * rows * dtype_size * bands


def guard_allocation(nbytes: int, *, limit: int = MAX_WINDOW_BYTES) -> None:
    """Refuse an allocation that would threaten the host.

    Raising here is far kinder than the alternative: on a 10 GB machine an
    oversized read does not fail cleanly, it drives the whole system into swap.
    """
    if nbytes > limit:
        raise WindowTooLargeError(
            f"refusing to allocate {nbytes / 1e6:.0f} MB "
            f"(limit {limit / 1e6:.0f} MB). Read a smaller window, or tile it."
        )


def clamp_window(window: PixelWindow, raster_width: int, raster_height: int) -> PixelWindow:
    """Clip a window to the raster, so an AOI overhanging the scene is safe."""
    col_off = max(0, min(window.col_off, raster_width))
    row_off = max(0, min(window.row_off, raster_height))
    width = max(0, min(window.width + min(0, window.col_off), raster_width - col_off))
    height = max(0, min(window.height + min(0, window.row_off), raster_height - row_off))
    return PixelWindow(col_off, row_off, width, height)


def tile_window(
    window: PixelWindow, tile_size: int = 2048, overlap: int = 512
) -> list[PixelWindow]:
    """Split a window into overlapping tiles.

    2048 with 512 overlap (stride 1536) is the xView3-winning scheme. The
    overlap removes tile-boundary artefacts and doubles as free test-time
    augmentation; detections are accumulated at scene level and reduced once,
    never merged per tile.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile_size")
    if window.is_empty():
        return []

    stride = tile_size - overlap
    tiles: list[PixelWindow] = []
    for row in range(window.row_off, window.row_off + window.height, stride):
        for col in range(window.col_off, window.col_off + window.width, stride):
            width = min(tile_size, window.col_off + window.width - col)
            height = min(tile_size, window.row_off + window.height - row)
            if width > 0 and height > 0:
                tiles.append(PixelWindow(col, row, width, height))
    return tiles


def bbox_to_window(dataset: Any, bbox: BBox) -> PixelWindow:
    """Convert a geographic bbox to a pixel window on an open dataset.

    Handles map-projected rasters and Sentinel-1 GRD scenes alike. GRD scenes
    carry ground control points instead of a geotransform: an earlier version
    read ``dataset.crs`` and ``dataset.transform`` directly, and on a real
    scene those are ``None`` and the identity, so it could not locate a single
    pixel. See :mod:`core.sar.georef`.
    """
    from core.sar.georef import SceneGeoref

    return SceneGeoref.from_dataset(dataset).window_for_bbox(bbox)


def read_window(dataset: Any, window: PixelWindow, *, band: int = 1) -> np.ndarray:
    """Read one window, refusing anything that would exhaust memory."""
    from rasterio.windows import Window

    if window.is_empty():
        raise ValueError("cannot read an empty window")
    guard_allocation(
        window.nbytes(
            dtype_size=dataset.dtypes[band - 1].itemsize
            if hasattr(dataset.dtypes[band - 1], "itemsize")
            else 4
        )
    )
    return dataset.read(
        band, window=Window(window.col_off, window.row_off, window.width, window.height)
    )


# Sigmoid calibration. Solved so that open ocean (~80 DN at VV) maps to about
# 0.1 and a bright ship return (~1200 DN) to about 0.9, putting the responsive
# part of the curve exactly where vessels are instead of spending it on the
# huge dynamic range of empty water.
SIGMOID_SCALE = 0.004
SIGMOID_OFFSET = 640.0


def normalise_sar(
    array: np.ndarray,
    *,
    scale: float = SIGMOID_SCALE,
    offset: float = SIGMOID_OFFSET,
) -> np.ndarray:
    """Map raw SAR amplitude into [0, 1] with a tuned sigmoid.

    Not linear scaling. The xView3-winning approach uses a sigmoid so that sea
    and land saturate towards 0 and 1 while the neighbourhood of a ship lands
    in the responsive middle of the curve, which measurably beat naive linear
    normalisation.

    Whatever is chosen here must be FROZEN: normalisation is a silent source of
    train/serve skew, and a model trained under one mapping will quietly
    degrade under another.
    """
    import numpy as np

    values = array.astype(np.float32, copy=False)
    # Missing data is encoded as 0 in GRD products and must stay at 0.
    missing = values == 0
    result = 1.0 / (1.0 + np.exp(-scale * (values - offset)))
    result[missing] = 0.0
    return result
