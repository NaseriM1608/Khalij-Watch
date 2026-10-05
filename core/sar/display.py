"""Rendering SAR amplitude for people to look at."""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from core.geo import EARTH_RADIUS_M

if TYPE_CHECKING:  # pragma: no cover
    from core.sar.georef import SceneGeoref
    from core.sar.windows import PixelWindow

# Raster pixels read beyond the ones a chip samples, so every sample has its
# four neighbours.
WINDOW_PAD_PX = 2
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def display_stretch(pixels: np.ndarray) -> np.ndarray:
    """Percentile stretch for viewing only.

    Deliberately not the frozen sigmoid used as model input: this exists to
    make faint vessels visible to a person, and must never feed a model.
    """
    valid = pixels[pixels > 0]
    if valid.size == 0:
        return np.zeros(pixels.shape, dtype=np.uint8)
    low, high = np.percentile(valid, [2.0, 99.8])
    scaled = np.clip((pixels.astype(np.float32) - low) / max(high - low, 1.0), 0.0, 1.0)
    return (np.sqrt(scaled) * 255.0).astype(np.uint8)


def north_up_grid(
    lon: float, lat: float, *, size_px: int, pixel_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of the pixel centres of a north-up square image centred on a point.

    Row 0 is the northern edge and column 0 the western; pixels are
    ``pixel_m`` on a side on a local tangent plane of the sphere, exact enough
    over a few kilometres.
    """
    offsets = (np.arange(size_px, dtype=np.float64) + 0.5 - size_px / 2.0) * pixel_m
    east, north = np.meshgrid(offsets, -offsets)
    lats = lat + np.degrees(north / EARTH_RADIUS_M)
    lons = lon + np.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lons, lats


def sample_bilinear(pixels: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``pixels`` at continuous indices (row 2.0 is the centre of row 2).

    Zero is no data in GRD products, so a sample outside the array, or with a
    zero among its four neighbours, is 0: half a missing pixel is not a darker
    pixel. Needs at least 2 x 2 pixels.
    """
    values = np.asarray(pixels, dtype=np.float64)
    height, width = values.shape
    if height < 2 or width < 2:
        raise ValueError(f"bilinear sampling needs at least 2 x 2 pixels, got {width} x {height}")
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    inside = (rows >= 0) & (rows <= height - 1) & (cols >= 0) & (cols <= width - 1)
    r = np.where(inside, rows, 0.0)
    c = np.where(inside, cols, 0.0)
    r0 = np.minimum(np.floor(r).astype(np.intp), height - 2)
    c0 = np.minimum(np.floor(c).astype(np.intp), width - 2)
    fr, fc = r - r0, c - c0
    p00, p01 = values[r0, c0], values[r0, c0 + 1]
    p10, p11 = values[r0 + 1, c0], values[r0 + 1, c0 + 1]
    blended = (1 - fr) * ((1 - fc) * p00 + fc * p01) + fr * ((1 - fc) * p10 + fc * p11)
    valid = inside & (p00 > 0) & (p01 > 0) & (p10 > 0) & (p11 > 0)
    return np.where(valid, blended, 0.0)


def db_stretch(pixels: np.ndarray, *, low_dn: float, high_dn: float) -> np.ndarray:
    """Amplitude to 8-bit grey on a fixed logarithmic (dB) scale.

    ``low_dn`` and below give 1, ``high_dn`` and above 255; no data (0) stays
    0. Fixed, unlike :func:`display_stretch`: the same amplitude gives the same
    grey in every chip, so a faint return next to a bright ship stays faint.
    For viewing only; never a model input.
    """
    if not 0 < low_dn < high_dn:
        raise ValueError(f"need 0 < low_dn < high_dn, got {low_dn}, {high_dn}")
    values = np.asarray(pixels, dtype=np.float64)
    valid = values > 0
    scaled = np.log(np.where(valid, values, low_dn) / low_dn) / math.log(high_dn / low_dn)
    grey = 1.0 + np.round(254.0 * np.clip(scaled, 0.0, 1.0))
    return np.where(valid, grey, 0.0).astype(np.uint8)


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def encode_gray_png(gray: np.ndarray, *, transparent_zero: bool = False) -> bytes:
    """An 8-bit greyscale PNG of a 2-D ``uint8`` array, written without GDAL.

    With ``transparent_zero``, grey 0 (no data, as :func:`db_stretch` keeps it)
    is transparent, so a viewer can show where the radar did not look rather
    than a black that reads as calm sea.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or 0 in gray.shape:
        raise ValueError(f"need a non-empty 2-D uint8 array, got {gray.dtype} {gray.shape}")
    height, width = gray.shape
    scanlines = np.zeros((height, width + 1), dtype=np.uint8)  # filter byte 0: none
    scanlines[:, 1:] = gray
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    transparency = _png_chunk(b"tRNS", struct.pack(">H", 0)) if transparent_zero else b""
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + transparency
        + _png_chunk(b"IDAT", zlib.compress(scanlines.tobytes(), 9))
        + _png_chunk(b"IEND", b"")
    )


def _covering_window(
    rows: np.ndarray, cols: np.ndarray, width: int, height: int
) -> PixelWindow | None:
    """The raster window holding every sample's neighbours, clipped to the raster; None if empty."""
    from core.sar.windows import PixelWindow

    finite = np.isfinite(rows) & np.isfinite(cols)
    if not finite.any():
        return None
    row0 = max(math.floor(float(rows[finite].min())) - WINDOW_PAD_PX, 0)
    row1 = min(math.ceil(float(rows[finite].max())) + WINDOW_PAD_PX + 1, height)
    col0 = max(math.floor(float(cols[finite].min())) - WINDOW_PAD_PX, 0)
    col1 = min(math.ceil(float(cols[finite].max())) + WINDOW_PAD_PX + 1, width)
    if row1 - row0 < 2 or col1 - col0 < 2:
        return None
    return PixelWindow(col0, row0, col1 - col0, row1 - row0)


def cut_north_up(
    datasets: Mapping[str, Any],
    georef: SceneGeoref,
    lon: float,
    lat: float,
    *,
    size_px: int,
    pixel_m: float,
    read: Callable[[Any, PixelWindow], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Amplitude (DN) of a north-up square image around a point, per polarisation.

    Ground-range scenes are in radar geometry: a descending pass shows the sea
    mirrored east-west and turned by the track angle. Each output pixel is
    located in the raster through the scene's control points and sampled
    bilinearly from one small window per band, so the chip reads like the map.
    ``datasets`` maps a polarisation to an open band in ``georef``'s geometry.
    Pixels beyond the scene's edge are 0 (no data); a square that misses the
    raster altogether raises ValueError.
    """
    if read is None:
        from core.sar.reader import read_with_retry as read

    lons, lats = north_up_grid(lon, lat, size_px=size_px, pixel_m=pixel_m)
    corner_rows, corner_cols = georef.raster_coords(lons.ravel(), lats.ravel())
    # raster_coords counts from pixel corners; sampling indexes pixel centres.
    rows, cols = corner_rows - 0.5, corner_cols - 0.5
    window = _covering_window(rows, cols, georef.width, georef.height)
    if window is None:
        raise ValueError(f"the scene does not cover {lon:.5f}, {lat:.5f}")
    return {
        pol: sample_bilinear(
            read(dataset, window), rows - window.row_off, cols - window.col_off
        ).reshape(size_px, size_px)
        for pol, dataset in datasets.items()
    }
