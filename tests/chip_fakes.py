"""Cutting radar chips without object storage: a real control-point grid, synthetic pixels.

The grid is a real Sentinel-1 one - S1D IW GRDH, acquired 2026-09-10 on a
descending pass over the Gulf of Oman, the pass of the console's dark-vessel
candidates - so chip geometry is tested on genuine radar geometry. Contains
modified Copernicus Sentinel data 2026. Import after
``pytest.importorskip("rasterio")``.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

FIXTURE = Path(__file__).parent / "fixtures" / "s1d_20260910T020640_gcps.json"
SCENE = "S1D_IW_GRDH_1SDV_20260910T020640_20260910T020705_004509_00860C_1202_COG.SAFE"
# A dark-vessel candidate of this very pass (data/khalij.db, run fusion-v1).
LON, LAT = 56.58012, 25.28648
SEA_DN, SHIP_DN = 60, 3000


def load_grid() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def load_gcps(grid: dict) -> tuple[GroundControlPoint, ...]:
    return tuple(
        GroundControlPoint(row=row, col=col, x=x, y=y, z=z) for row, col, x, y, z in grid["gcps"]
    )


class FakeBand:
    """A whole-scene raster rendered window by window: sea, plus bright pixels where asked."""

    def __init__(self, width: int, height: int, *, sea: int = SEA_DN, targets=None, gcps=()):
        self.width, self.height = width, height
        self.dtypes = ("uint16",)
        self.gcps = (list(gcps), CRS.from_epsg(4326) if gcps else None)
        self.sea = sea
        self.targets = dict(targets or {})
        self.windows = []

    def read(self, _band, window=None):
        self.windows.append(window)
        row0, col0 = int(window.row_off), int(window.col_off)
        height, width = int(window.height), int(window.width)
        out = np.full((height, width), self.sea, dtype=np.uint16)
        for (row, col), value in self.targets.items():
            if row0 <= row < row0 + height and col0 <= col < col0 + width:
                out[row - row0, col - col0] = value
        return out


def decode_gray_png(data: bytes) -> tuple[np.ndarray, int | None]:
    """Pixels of an 8-bit greyscale PNG, and the grey value its tRNS chunk makes transparent."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(kind + body)
        chunks[kind] = chunks.get(kind, b"") + body
        pos += 12 + length
    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, colour) == (8, 0)
    raw = zlib.decompress(chunks[b"IDAT"])
    rows = [raw[r * (width + 1) : (r + 1) * (width + 1)] for r in range(height)]
    assert all(row[0] == 0 for row in rows)
    assert b"IEND" in chunks
    transparent = struct.unpack(">H", chunks[b"tRNS"])[0] if b"tRNS" in chunks else None
    return np.array([list(row[1:]) for row in rows], dtype=np.uint8), transparent
