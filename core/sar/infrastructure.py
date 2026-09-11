"""Offshore infrastructure mask for the Persian Gulf.

The Gulf is dense with oil and gas platforms - 2,314 distinct structures in
the 2017-2025 record - and every one is a bright, permanent point target. To a
CFAR they look exactly like a stationary vessel, so without this mask they are
the dominant false-alarm source over water, and no amount of threshold tuning
removes them.

Source: the Offshore Platform Dataset (OPD v1.0.0), derived from the Sentinel-1
archive by deep-learning object detection.
    Spanier, Hoeser, Truckenbrodt, Bachofer & Kuenzer (2026), Big Earth Data.
    https://doi.org/10.5281/zenodo.19046824    CC BY 4.0

Why the mask is time-aware
--------------------------
Platforms are installed and decommissioned throughout the record. A structure
first seen in 2023Q4 must not mask a 2019 scene - there was nothing there, and
masking it would hide a real vessel. Equally, one last seen in 2021 must not
mask a 2024 scene. Each entry therefore carries a lifespan and is only applied
to imagery acquired inside it.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

from core.db.spatial import BBox

DEFAULT_PATH = Path("data") / "reference" / "OPD_platforms.geojson"
PERSIAN_GULF = "Persian Gulf"

ATTRIBUTION = (
    "Offshore Platform Dataset (OPD v1.0.0), Spanier et al. 2026, "
    "doi:10.5281/zenodo.19046824, CC BY 4.0"
)

# Radius masked around each platform. The median detected footprint is about
# 145 m across, but structures have guard vessels, flare booms and mooring
# spreads alongside, and geolocation is not perfect. A slightly generous
# radius costs a little sea area and saves a great many false alarms.
DEFAULT_RADIUS_M = 250.0

# Open-ended lifespan markers used by the source dataset.
_INSTALLED_BEFORE_RECORD = "17Q1 or before"
_STILL_PRESENT = "25Q1 or still present"

_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQUATOR = 111_320.0


def parse_quarter(text: str | None) -> dt.date | None:
    """Parse a ``YYQn`` marker into the first day of that quarter.

    Returns ``None`` for the dataset's open-ended markers, which mean "before
    the record began" or "still present" rather than a specific date.
    """
    if not text or text in (_INSTALLED_BEFORE_RECORD, _STILL_PRESENT):
        return None
    cleaned = text.strip().upper()
    if len(cleaned) != 4 or cleaned[2] != "Q":
        return None
    try:
        year = 2000 + int(cleaned[:2])
        quarter = int(cleaned[3])
    except ValueError:
        return None
    if not 1 <= quarter <= 4:
        return None
    return dt.date(year, 3 * (quarter - 1) + 1, 1)


def quarter_end(start: dt.date) -> dt.date:
    """Last day of the quarter beginning at ``start``."""
    month = start.month + 3
    year = start.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return dt.date(year, month, 1) - dt.timedelta(days=1)


@dataclass(frozen=True)
class Platform:
    """One fixed offshore structure."""

    lon: float
    lat: float
    installed: dt.date | None
    last_observed: dt.date | None
    country_eez: str | None = None
    water_depth_m: float | None = None
    footprint_ha: float | None = None

    def was_present(self, when: dt.date) -> bool:
        """Whether the structure existed at a given date.

        ``None`` bounds are open: installed before the record, or still
        standing at its end.
        """
        if self.installed is not None and when < self.installed:
            return False
        return not (self.last_observed is not None and when > quarter_end(self.last_observed))


def _feature_to_platform(feature: dict) -> Platform | None:
    props = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates")
    if geometry.get("type") != "Point" or not coords or len(coords) < 2:
        return None
    return Platform(
        lon=float(coords[0]),
        lat=float(coords[1]),
        installed=parse_quarter(props.get("installed_quarter")),
        last_observed=parse_quarter(props.get("last_observed_quarter")),
        country_eez=props.get("country_eez"),
        water_depth_m=props.get("water_depth_m"),
        footprint_ha=props.get("detection_bbox_area_ha"),
    )


def load_platforms(
    path: Path | None = None,
    *,
    region: str | None = PERSIAN_GULF,
    bbox: BBox | None = None,
) -> list[Platform]:
    """Load platforms from an OPD GeoJSON file.

    Raises :class:`FileNotFoundError` with an actionable message rather than a
    bare path, because the fix is a single script away.
    """
    path = path or DEFAULT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with: uv run python scripts/fetch_platforms.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    platforms: list[Platform] = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        if region is not None and props.get("region") != region:
            continue
        platform = _feature_to_platform(feature)
        if platform is None:
            continue
        if bbox is not None and not bbox.contains(platform.lon, platform.lat):
            continue
        platforms.append(platform)
    return platforms


def active_platforms(platforms: list[Platform], when: dt.date) -> list[Platform]:
    """Filter to structures standing on a given date."""
    return [p for p in platforms if p.was_present(when)]


class InfrastructureMask:
    """Rasterises platform positions into a per-window exclusion mask."""

    def __init__(self, platforms: list[Platform], *, radius_m: float = DEFAULT_RADIUS_M) -> None:
        self._platforms = platforms
        self._radius_m = radius_m

    def __len__(self) -> int:
        return len(self._platforms)

    @property
    def platforms(self) -> list[Platform]:
        return self._platforms

    def for_date(self, when: dt.date) -> InfrastructureMask:
        """A mask containing only structures standing on that date."""
        return InfrastructureMask(active_platforms(self._platforms, when), radius_m=self._radius_m)

    def covers(self, lon: float, lat: float) -> bool:
        """Whether a position falls within the masked radius of a structure."""
        for platform in self._platforms:
            if _metres_between(lon, lat, platform.lon, platform.lat) <= self._radius_m:
                return True
        return False

    def rasterise(
        self,
        georef: object,
        width: int,
        height: int,
        *,
        geographic: bool | None = None,
        window_offset: tuple[int, int] = (0, 0),
    ):
        """Build a boolean exclusion mask for a raster window.

        ``georef`` is a :class:`core.sar.georef.SceneGeoref` - required for
        Sentinel-1 GRD scenes, which are located by ground control points
        rather than a geotransform - or an affine transform for a north-up
        lon/lat raster. ``window_offset`` is the (row, col) origin of this
        window within the scene. ``geographic`` applies only to the affine
        form, where it says whether the transform is in degrees (see
        :meth:`_radius_in_pixels`).
        """
        import numpy as np

        from core.sar.georef import SceneGeoref

        mask = np.zeros((height, width), dtype=bool)
        if not self._platforms:
            return mask

        row_off, col_off = window_offset
        if isinstance(georef, SceneGeoref):
            scene = georef
            # The finer of the two pixel spacings, so the disc is never too small.
            spacing = min(scene.metres_per_pixel(row_off + height / 2.0, col_off + width / 2.0))
            radii = [self._radius_m / max(spacing, 1e-6)] * len(self._platforms)
        else:
            scene = SceneGeoref.from_affine(georef)
            radii = [
                self._radius_in_pixels(georef, p.lat, geographic=geographic)
                for p in self._platforms
            ]

        rows, cols = scene.raster_coords(
            [p.lon for p in self._platforms], [p.lat for p in self._platforms]
        )
        for row, col, radius_px in zip(rows, cols, radii, strict=True):
            if not (math.isfinite(row) and math.isfinite(col)):
                continue
            # raster_coords is corner-based, so a pixel-index centre sits half
            # a pixel back; then shift into this window's own frame.
            _stamp_disc(mask, col - 0.5 - col_off, row - 0.5 - row_off, radius_px)
        return mask

    def _radius_in_pixels(self, transform, lat: float, *, geographic: bool | None = None) -> float:
        """Convert the mask radius from metres to pixels.

        ``transform.a`` is the west-east pixel size in whatever units the CRS
        uses. When the caller does not say which, it is inferred: no projected
        raster has metre-scale pixels below 1.0, and no geographic raster has
        degree-scale pixels above it. An earlier version used 0.01 as the
        cutoff, which misread a 0.01-degree pixel as 0.01 metres and inflated
        the radius by four orders of magnitude.
        """
        pixel_size = abs(transform.a)
        if geographic is None:
            geographic = pixel_size < 1.0
        if geographic:
            metres_per_pixel = pixel_size * _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat))
        else:
            metres_per_pixel = pixel_size
        return self._radius_m / max(metres_per_pixel, 1e-6)


def _stamp_disc(mask, col: float, row: float, radius_px: float) -> None:
    """Set a filled circle in a boolean mask, clipped to its bounds."""
    import numpy as np

    height, width = mask.shape
    radius = max(radius_px, 0.5)
    r0 = max(0, math.floor(row - radius))
    r1 = min(height, math.ceil(row + radius) + 1)
    c0 = max(0, math.floor(col - radius))
    c1 = min(width, math.ceil(col + radius) + 1)
    if r0 >= r1 or c0 >= c1:
        return

    rows = np.arange(r0, r1)[:, None]
    cols = np.arange(c0, c1)[None, :]
    inside = (rows - row) ** 2 + (cols - col) ** 2 <= radius**2
    mask[r0:r1, c0:c1] |= inside


def _metres_between(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Local planar distance. Adequate at the few-hundred-metre scale here."""
    mid_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * _M_PER_DEG_LON_EQUATOR * math.cos(mid_lat)
    dy = (lat2 - lat1) * _M_PER_DEG_LAT
    return math.hypot(dx, dy)
