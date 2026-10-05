"""GFW hourly AIS presence around a Sentinel-1 pass, as per-vessel hourly tracks.

The only free AIS at our pass times is Global Fishing Watch's 4Wings presence
report (``public-global-presence`` v4.0) at HIGH resolution (0.01 degree
cells, read as centres), HOURLY, grouped by VESSEL_ID. Measured 2026-10-04
(PLAN §17 "Phase 6 feasibility"): one row per vessel per hour, each worth
exactly 1 hour; no speed or course; entry and exit times truncated to the
hour; hour-level query windows honoured, end exclusive. A vessel at a pass is
therefore best described by the polyline through its cells in the hour
before, the hour of and the hour after the pass (H-1, H, H+1). At the busy
Fujairah pass of 2026-09-25, 91 of the 127 vessels present in all three hours
stayed in one cell.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import Any

from core.db.spatial import BBox
from core.fusion.geometry import local_distance_m
from core.labeling.gfw import _as_int, _floor_hour, _parse_hour
from core.labeling.gfw_api import box_polygon, with_retries

# H-1, H and H+1 around the pass hour H.
WINDOW_HOURS = 3
_HOUR = dt.timedelta(hours=1)


class _GfwHour(dt.datetime):
    """A UTC hour that gfwapiclient renders in the form 4Wings accepts, e.g. '2026-09-25T13:00:00.000Z'.

    gfwapiclient 1.4 builds the date-range from ``date.isoformat()``. 4Wings
    rejects '2026-09-25T13:00:00' (HTTP 422) but honours '.000Z' hour bounds:
    through the client this reproduced P10's 422 vessel-hours exactly
    (measured 2026-10-04).
    """

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:  # noqa: ARG002
        return self.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    @classmethod
    def of(cls, moment: dt.datetime) -> _GfwHour:
        utc = moment.astimezone(dt.UTC)
        return cls(utc.year, utc.month, utc.day, utc.hour, tzinfo=dt.UTC)


@dataclass(frozen=True)
class PresenceFix:
    """One vessel-hour: the centre of the 0.01 degree cell the vessel was in during one UTC hour."""

    vessel_id: str
    hour: dt.datetime
    lon: float
    lat: float
    mmsi: int | None = None
    imo: str | None = None
    ship_name: str | None = None
    flag: str | None = None
    vessel_type: str | None = None
    # The served dataset version, e.g. 'public-global-presence:v4.0' (requests ask for ':latest').
    report_dataset: str | None = None


@dataclass(frozen=True)
class HourlyTrack:
    """A vessel's fixes in the hours H-1, H and H+1 around a pass, ordered by hour.

    ``pass_hour`` is H; ``pass_fraction`` is how far into H the pass was (0..1).
    """

    vessel_id: str
    fixes: tuple[PresenceFix, ...]
    pass_hour: dt.datetime | None = None
    pass_fraction: float = 0.0

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return [(f.lon, f.lat) for f in self.fixes]

    @property
    def gate_vertices(self) -> list[tuple[float, float]]:
        """The polyline the gate measures to, reaching the pass time.

        A fix in hour H lies somewhere in [H, H+1). A vessel seen only at H-1
        and H can be up to ``pass_fraction`` of an hour of travel beyond its H
        cell at the pass, and one seen only at H and H+1 up to the rest of the
        hour before it, so those tracks are extended along their direction.
        Tracks with H-1 and H+1 already span the pass; single fixes cannot be
        extended.
        """
        vertices = self.vertices
        if self.pass_hour is None or len(self.fixes) != 2:
            return vertices
        offsets = [f.hour - self.pass_hour for f in self.fixes]
        (x0, y0), (x1, y1) = vertices
        if offsets == [-_HOUR, dt.timedelta(0)]:
            f = self.pass_fraction
            return [*vertices, (x1 + f * (x1 - x0), y1 + f * (y1 - y0))]
        if offsets == [dt.timedelta(0), _HOUR]:
            g = 1.0 - self.pass_fraction
            return [(x0 - g * (x1 - x0), y0 - g * (y1 - y0)), *vertices]
        return vertices

    @property
    def sparse(self) -> bool:
        """Seen in only one of the three hours, so its speed is unknown."""
        return len(self.fixes) == 1

    @property
    def stationary(self) -> bool:
        """In one cell in all three hours: anchored or moored."""
        return len(self.fixes) == WINDOW_HOURS and len({(f.lon, f.lat) for f in self.fixes}) == 1

    @property
    def speed_mps(self) -> float | None:
        """The fastest cell-to-cell displacement per elapsed time; None for a single fix.

        An estimate that can err either way: a vessel moving less than a cell
        an hour shows zero, and the sampled minute within each hour is
        unknown, so two fixes can be less or more than an hour apart in time.
        """
        if self.sparse:
            return None
        return max(
            local_distance_m(a.lon, a.lat, b.lon, b.lat) / (b.hour - a.hour).total_seconds()
            for a, b in pairwise(self.fixes)
        )


def parse_presence_rows(rows: Iterable[dict]) -> list[PresenceFix]:
    """Fixes from presence rows (gfwapiclient ``model_dump()`` names), dropping rows without vessel, cell or hour."""
    fixes: list[PresenceFix] = []
    for row in rows:
        vessel_id, lat, lon = row.get("vessel_id"), row.get("lat"), row.get("lon")
        hour = _parse_hour(row.get("date"))
        if not vessel_id or lat is None or lon is None or hour is None:
            continue
        fixes.append(
            PresenceFix(
                vessel_id=str(vessel_id),
                hour=hour,
                lon=float(lon),
                lat=float(lat),
                mmsi=_as_int(row.get("mmsi")),
                imo=row.get("imo") or None,
                ship_name=row.get("ship_name") or None,
                flag=row.get("flag") or None,
                vessel_type=row.get("vessel_type") or None,
                report_dataset=row.get("report_dataset") or None,
            )
        )
    return fixes


def presence_window(acquired: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    """The UTC hours H-1 .. H+1 around a pass acquired at ``acquired``, as [start, end)."""
    if acquired.tzinfo is None:
        raise ValueError("acquired must carry a timezone (UTC)")
    hour = _floor_hour(acquired)
    return hour - dt.timedelta(hours=1), hour + dt.timedelta(hours=WINDOW_HOURS - 1)


def hourly_tracks(fixes: Iterable[PresenceFix], acquired: dt.datetime) -> list[HourlyTrack]:
    """One track per vessel from its fixes inside the pass window (the first fix of an hour wins)."""
    start, end = presence_window(acquired)
    pass_hour = start + _HOUR
    fraction = (acquired - pass_hour).total_seconds() / _HOUR.total_seconds()
    by_vessel: dict[str, dict[dt.datetime, PresenceFix]] = defaultdict(dict)
    for f in fixes:
        if start <= f.hour < end:
            by_vessel[f.vessel_id].setdefault(f.hour, f)
    return [
        HourlyTrack(
            vessel_id=vessel_id,
            fixes=tuple(hours[h] for h in sorted(hours)),
            pass_hour=pass_hour,
            pass_fraction=fraction,
        )
        for vessel_id, hours in sorted(by_vessel.items())
    ]


def failed_before_sending(exc: Exception) -> bool:
    """Whether a client error happened before the report reached GFW, so it is safe to send again.

    gfwapiclient wraps every send failure (``raise ... from`` the httpx error).
    After the request has left - a read timeout, a dropped response - GFW may
    still be computing the report, and sending it again would only meet the
    one-report lock (HTTP 429).
    """
    import httpx

    return isinstance(exc.__cause__, httpx.ConnectError | httpx.ConnectTimeout | httpx.PoolTimeout)


async def fetch_presence(
    client: Any,
    box: BBox,
    acquired: dt.datetime,
    *,
    transient: Callable[[Exception], bool] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> list[PresenceFix]:
    """GFW AIS presence (HIGH, HOURLY, by vessel) inside ``box`` for the hours H-1..H+1 around a pass.

    One 4Wings report: only one may run at a time per token, so callers fetch
    passes one after another. Pass the box buffered by about an hour of travel.
    """
    start, end = presence_window(acquired)
    request = partial(
        client.fourwings.create_ais_presence_report,
        spatial_resolution="HIGH",
        temporal_resolution="HOURLY",
        group_by="VESSEL_ID",
        start_date=_GfwHour.of(start),
        end_date=_GfwHour.of(end),
        geojson=box_polygon(box),
    )
    result = await with_retries(
        request, is_transient=transient or failed_before_sending, sleep=sleep
    )
    return parse_presence_rows(row.model_dump() for row in result.data())
