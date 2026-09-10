"""Copernicus Data Space (CDSE) product catalogue client.

Searching the catalogue needs **no authentication** - only downloading pixels
does. So scene discovery works before any credentials exist, which is why this
module is deliberately separate from :mod:`core.sar.auth`.

Product naming, e.g.::

    S1D_IW_GRDH_1SDV_20260910T020615_20260910T020640_004509_00860C_4361.SAFE
    |0| |1| |2 | |3 | |------ 4 ------| |------ 5 ------|
     0 satellite   S1A / S1C / S1D
     1 mode        IW (the only mode used over the Gulf)
     2 product     GRDH / GRDM / SLC / OCN
     3 polarisation 1SDV = dual VV+VH, 1SSV = single VV
     4 start time  5 stop time

A name ending ``_COG.SAFE`` is the Cloud-Optimized GeoTIFF variant. That is the
one to use: it supports byte-range reads, so an AOI window can be fetched
without downloading the whole scene.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from core.db.spatial import BBox

ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
SENTINEL1 = "SENTINEL-1"

# Interferometric Wide swath, ground-range detected, high resolution. This is
# what Sentinel-1 acquires over land and coastal areas, so every Gulf scene is
# one of these. SLC is 4x larger, complex-valued, and only needed for
# interferometry or Doppler work.
PRODUCT_TYPE_GRDH = "IW_GRDH_1S"

DEFAULT_TIMEOUT = 90
DEFAULT_PAGE_SIZE = 1000
# The catalogue caps a page and hands back an @odata.nextLink. Ignoring that
# link truncates results SILENTLY, which for a discovery worker means missing
# scenes and reporting a coverage picture that is quietly wrong.
MAX_PAGES = 20
# The catalogue is public, but a default Python user-agent is sometimes
# refused by the fronting CDN.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class CatalogueError(RuntimeError):
    """The catalogue could not be queried or returned something unusable."""


@dataclass(frozen=True)
class SarProduct:
    """One catalogued Sentinel-1 product."""

    product_id: str
    name: str
    satellite: str
    mode: str
    product_type: str
    polarisation: str
    acq_start: dt.datetime
    acq_end: dt.datetime
    footprint: list[tuple[float, float]]
    bbox: BBox
    size_bytes: int
    s3_path: str | None
    orbit_direction: str | None
    timeliness: str | None
    online: bool
    is_cog: bool

    @property
    def is_dual_pol(self) -> bool:
        """VV+VH. VH is the better ship-detection channel: lower ocean
        backscatter means higher target-to-clutter contrast."""
        return self.polarisation == "DV"

    def age(self, now: dt.datetime | None = None) -> dt.timedelta:
        return (now or dt.datetime.now(dt.UTC)) - self.acq_start


def _parse_name(name: str) -> dict[str, str]:
    """Pull the structured fields out of a product name."""
    stem = name.removesuffix(".SAFE")
    is_cog = stem.endswith("_COG")
    parts = stem.split("_")
    if len(parts) < 4:
        raise CatalogueError(f"unparseable product name: {name!r}")
    # GRDH appears as "IW_GRDH_1S..." so mode and type are separate tokens.
    polarisation_token = parts[3]
    return {
        "satellite": parts[0],
        "mode": parts[1],
        "product_type": parts[2],
        # "1SDV" -> "DV" (dual VV+VH); "1SSV" -> "SV" (single VV)
        "polarisation": polarisation_token[2:4] if len(polarisation_token) >= 4 else "",
        "is_cog": "true" if is_cog else "",
    }


def _parse_time(raw: str) -> dt.datetime:
    cleaned = raw.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(cleaned)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _footprint_ring(geo: Any) -> list[tuple[float, float]]:
    """Extract the outer ring from a GeoJSON footprint."""
    if not isinstance(geo, dict):
        return []
    coords = geo.get("coordinates")
    if geo.get("type") == "Polygon" and coords:
        return [(float(x), float(y)) for x, y in coords[0]]
    if geo.get("type") == "MultiPolygon" and coords:
        return [(float(x), float(y)) for x, y in coords[0][0]]
    return []


def _bbox_of(ring: list[tuple[float, float]]) -> BBox:
    if not ring:
        raise CatalogueError("product has no usable footprint")
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return BBox(min(lons), min(lats), max(lons), max(lats))


def _attributes(raw: Any) -> dict[str, str]:
    if not isinstance(raw, list):
        return {}
    return {
        str(a.get("Name")): str(a.get("Value"))
        for a in raw
        if isinstance(a, dict) and a.get("Name") is not None
    }


def product_from_odata(entry: dict[str, Any]) -> SarProduct:
    """Build a :class:`SarProduct` from one OData record."""
    name = entry.get("Name")
    if not isinstance(name, str):
        raise CatalogueError("product record has no Name")

    fields = _parse_name(name)
    content = entry.get("ContentDate") or {}
    attributes = _attributes(entry.get("Attributes"))
    ring = _footprint_ring(entry.get("GeoFootprint"))

    return SarProduct(
        product_id=str(entry.get("Id", "")),
        name=name,
        satellite=fields["satellite"],
        mode=fields["mode"],
        product_type=fields["product_type"],
        polarisation=fields["polarisation"],
        acq_start=_parse_time(content.get("Start", "1970-01-01T00:00:00Z")),
        acq_end=_parse_time(content.get("End", content.get("Start", "1970-01-01T00:00:00Z"))),
        footprint=ring,
        bbox=_bbox_of(ring),
        size_bytes=int(entry.get("ContentLength") or 0),
        s3_path=entry.get("S3Path"),
        orbit_direction=attributes.get("orbitDirection"),
        timeliness=attributes.get("timeliness"),
        # Offline products live in long-term archive and need a retrieval order
        # before any pixels can be read.
        online=bool(entry.get("Online", True)),
        is_cog=bool(fields["is_cog"]),
    )


def _build_filter(
    bbox: BBox,
    since: dt.datetime,
    until: dt.datetime | None,
    product_type: str | None,
) -> str:
    # An intersects test against the AOI polygon, rather than a point, so
    # scenes clipping the edge of the area are still found.
    clauses = [
        f"Collection/Name eq '{SENTINEL1}'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox.as_wkt()}')",
        f"ContentDate/Start gt {since.strftime('%Y-%m-%dT%H:%M:%S.000Z')}",
    ]
    if until is not None:
        clauses.append(f"ContentDate/Start lt {until.strftime('%Y-%m-%dT%H:%M:%S.000Z')}")
    if product_type:
        clauses.append(f"contains(Name,'{product_type}')")
    return " and ".join(clauses)


def _fetch(url: str, opener=None) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=DEFAULT_TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise CatalogueError(f"catalogue returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CatalogueError(f"catalogue unreachable: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise CatalogueError(f"catalogue returned invalid JSON: {exc}") from exc


def search(
    bbox: BBox,
    *,
    since: dt.datetime,
    until: dt.datetime | None = None,
    product_type: str | None = PRODUCT_TYPE_GRDH,
    limit: int = DEFAULT_PAGE_SIZE,
    cog_only: bool = True,
    max_pages: int = MAX_PAGES,
    opener=None,
) -> list[SarProduct]:
    """Search the catalogue, following pagination. Requires no credentials.

    ``cog_only`` keeps only the Cloud-Optimized GeoTIFF variants, which are the
    ones that support windowed byte-range reads.
    """
    query = urllib.parse.urlencode(
        {
            "$filter": _build_filter(bbox, since, until, product_type),
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start desc",
            "$top": str(min(limit, 1000)),
        }
    )
    url: str | None = f"{ODATA_URL}?{query}"

    products: list[SarProduct] = []
    for _page in range(max_pages):
        if url is None:
            break
        payload = _fetch(url, opener)
        for entry in payload.get("value", []):
            try:
                product = product_from_odata(entry)
            except CatalogueError:
                # One malformed record must not lose the rest of the page.
                continue
            if cog_only and not product.is_cog:
                continue
            products.append(product)
        url = payload.get("@odata.nextLink")
    return products


def mean_revisit_days(products: list[SarProduct]) -> float:
    """Mean gap between distinct passes, in days.

    Products from a single pass share a start time within minutes, so they are
    collapsed: one pass is one observation.
    """
    stamps = sorted({p.acq_start.replace(microsecond=0) for p in products})
    gaps = [
        (b - a).total_seconds() / 86400
        for a, b in itertools.pairwise(stamps)
        if (b - a).total_seconds() > 3600
    ]
    return sum(gaps) / len(gaps) if gaps else 0.0


def parse_polygon_wkt(wkt: str | None) -> list[tuple[float, float]]:
    """Parse ``POLYGON((lon lat, ...))`` back into a coordinate ring.

    Only the simple single-ring form this project writes is supported; a full
    WKT parser would be a dependency for no gain.
    """
    if not wkt:
        return []
    text_value = wkt.strip().upper()
    if not text_value.startswith("POLYGON"):
        return []
    try:
        inner = wkt[wkt.index("((") + 2 : wkt.rindex("))")]
    except ValueError:
        return []
    ring: list[tuple[float, float]] = []
    for pair in inner.split(","):
        parts = pair.split()
        if len(parts) != 2:
            return []
        try:
            ring.append((float(parts[0]), float(parts[1])))
        except ValueError:
            return []
    return ring
