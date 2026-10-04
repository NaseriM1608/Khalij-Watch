"""Fetch Global Fishing Watch SAR presence cells for an area and period.

Uses the official gfw-api-python-client (Apache-2.0). GFW data are for
non-commercial use; the token is read from GFW_API_TOKEN by the caller.

The request shape was settled against the live API: HIGH spatial resolution
(0.01 degree cells), HOURLY time resolution, AIS-matched detections grouped
by VESSEL_ID (which returns identity fields; grouping by MMSI returns the MMSI
alone), and unmatched detections ungrouped, since they have no vessel.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from core.db.spatial import BBox
from core.labeling.gfw import GfwCell, parse_rows

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 5.0


def month_ranges(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Split ``[start, end)`` into pieces that never cross a calendar month."""
    if start >= end:
        raise ValueError(f"start {start} must be before end {end}")
    ranges: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor < end:
        next_month = (cursor.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        stop = min(next_month, end)
        ranges.append((cursor, stop))
        cursor = stop
    return ranges


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    *,
    is_transient: Callable[[Exception], bool],
    attempts: int = RETRY_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Await ``call``, retrying transient failures with a growing delay."""
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            await sleep(base_delay * attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def is_transient(exc: Exception) -> bool:
    """Connection drops and timeouts, which this host produces intermittently."""
    from gfwapiclient.exceptions import APIConnectionError, APITimeoutError

    return isinstance(exc, APIConnectionError | APITimeoutError)


def box_polygon(box: BBox) -> dict:
    """The GeoJSON polygon 4Wings reports take for a box."""
    ring = [
        [box.min_lon, box.min_lat],
        [box.max_lon, box.min_lat],
        [box.max_lon, box.max_lat],
        [box.min_lon, box.max_lat],
        [box.min_lon, box.min_lat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


async def fetch_sar_cells(
    client: Any,
    box: BBox,
    start: dt.date,
    end: dt.date,
    *,
    transient: Callable[[Exception], bool] | None = None,
) -> list[GfwCell]:
    """All matched and unmatched SAR presence cells inside ``box`` during ``[start, end)``."""
    polygon = box_polygon(box)
    transient = transient or is_transient
    cells: list[GfwCell] = []
    for first, stop in month_ranges(start, end):
        for matched, group_by in ((True, "VESSEL_ID"), (False, None)):
            request = partial(_report, client, polygon, first, stop, matched, group_by)
            result = await with_retries(request, is_transient=transient)
            rows = [row.model_dump() for row in result.data()]
            cells.extend(parse_rows(rows, matched=matched))
    return cells


async def _report(
    client: Any,
    polygon: dict,
    first: dt.date,
    stop: dt.date,
    matched: bool,
    group_by: str | None,
) -> Any:
    return await client.fourwings.create_sar_presence_report(
        spatial_resolution="HIGH",
        temporal_resolution="HOURLY",
        group_by=group_by,
        filters=[f"matched='{'true' if matched else 'false'}'"],
        start_date=first.isoformat(),
        end_date=stop.isoformat(),
        geojson=polygon,
    )
