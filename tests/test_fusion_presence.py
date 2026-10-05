"""GFW hourly AIS presence -> per-vessel hourly tracks around a pass.

GFW's only free AIS at our pass times is its 4Wings presence report: one row
per vessel per hour, positioned at a 0.01 degree cell centre, with no time
inside the hour (PLAN §17, 2026-10-04 "Phase 6 feasibility"). These tests pin
how rows become fixes and fixes become the H-1..H+1 tracks the gate uses.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.db.spatial import BBox
from core.fusion.presence import (
    PresenceFix,
    fetch_presence,
    hourly_tracks,
    parse_presence_rows,
    presence_window,
)

PASS = dt.datetime(2026, 9, 25, 14, 15, 51, tzinfo=dt.UTC)


def row(vessel_id="v1", date="2026-09-25 14:00", lat=25.20, lon=56.50, **extra):
    """A presence row as gfwapiclient's model_dump() gives it (field names, not aliases)."""
    return {
        "vessel_id": vessel_id,
        "date": date,
        "lat": lat,
        "lon": lon,
        "hours": 1,
        "mmsi": "538005255",
        "imo": "9387152",
        "ship_name": "ZENITH FAITH",
        "flag": "MHL",
        "vessel_type": "OTHER",
        **extra,
    }


def fix(vessel_id="v1", hour=14, lat=25.20, lon=56.50):
    return PresenceFix(
        vessel_id=vessel_id, hour=dt.datetime(2026, 9, 25, hour, tzinfo=dt.UTC), lon=lon, lat=lat
    )


# ---------------------------------------------------------------------------
# Rows to fixes
# ---------------------------------------------------------------------------


def test_rows_become_fixes_with_identity_and_utc_hours():
    (parsed,) = parse_presence_rows([row()])
    assert parsed.vessel_id == "v1"
    assert parsed.hour == dt.datetime(2026, 9, 25, 14, tzinfo=dt.UTC)
    assert (parsed.lon, parsed.lat) == (56.50, 25.20)
    assert parsed.mmsi == 538005255
    assert (parsed.imo, parsed.ship_name, parsed.flag, parsed.vessel_type) == (
        "9387152",
        "ZENITH FAITH",
        "MHL",
        "OTHER",
    )


def test_rows_that_cannot_be_placed_or_identified_are_dropped():
    rows = [
        row(vessel_id=""),
        row(lat=None),
        row(date="not a date"),
        row(date=None),
        row(vessel_id="ok"),
    ]
    assert [f.vessel_id for f in parse_presence_rows(rows)] == ["ok"]


def test_blank_identity_fields_become_none():
    (parsed,) = parse_presence_rows([row(mmsi="", imo="", ship_name="", flag="", vessel_type="")])
    assert (parsed.mmsi, parsed.imo, parsed.ship_name, parsed.flag, parsed.vessel_type) == (
        None,
    ) * 5


# ---------------------------------------------------------------------------
# The query window around a pass
# ---------------------------------------------------------------------------


def test_window_spans_the_hour_before_the_pass_hour_and_the_hour_after():
    start, end = presence_window(PASS)
    assert start == dt.datetime(2026, 9, 25, 13, tzinfo=dt.UTC)
    assert end == dt.datetime(2026, 9, 25, 16, tzinfo=dt.UTC)  # exclusive, as GFW treats it


def test_window_of_a_pass_on_the_hour_starts_an_hour_earlier():
    start, end = presence_window(dt.datetime(2026, 9, 25, 14, 0, 0, tzinfo=dt.UTC))
    assert (start.hour, end.hour) == (13, 16)


def test_window_rejects_naive_times():
    with pytest.raises(ValueError, match="timezone"):
        presence_window(dt.datetime(2026, 9, 25, 14, 15))


# ---------------------------------------------------------------------------
# Fixes to hourly tracks
# ---------------------------------------------------------------------------


def test_fixes_group_into_one_track_per_vessel_ordered_by_hour():
    tracks = hourly_tracks([fix("a", 15), fix("b", 14), fix("a", 13), fix("a", 14)], PASS)
    by_id = {t.vessel_id: t for t in tracks}
    assert sorted(by_id) == ["a", "b"]
    assert [f.hour.hour for f in by_id["a"].fixes] == [13, 14, 15]


def test_fixes_outside_the_window_are_left_out():
    tracks = hourly_tracks([fix("a", 12), fix("a", 14), fix("a", 16), fix("c", 11)], PASS)
    assert [(t.vessel_id, [f.hour.hour for f in t.fixes]) for t in tracks] == [("a", [14])]


def test_a_vessel_in_one_cell_for_three_hours_is_stationary_with_zero_speed():
    (track,) = hourly_tracks([fix(hour=h) for h in (13, 14, 15)], PASS)
    assert track.stationary
    assert track.speed_mps == 0.0
    assert not track.sparse


def test_speed_is_the_fastest_hourly_cell_displacement():
    # 0.1 deg of latitude in an hour is about 11 km/h, about 3.07 m/s.
    (track,) = hourly_tracks(
        [fix(hour=13, lat=25.0), fix(hour=14, lat=25.05), fix(hour=15, lat=25.15)], PASS
    )
    assert not track.stationary
    assert track.speed_mps == pytest.approx(0.10 * 110_540 / 3600, rel=1e-3)


def test_speed_over_a_missing_hour_uses_the_elapsed_time():
    (track,) = hourly_tracks([fix(hour=13, lat=25.0), fix(hour=15, lat=25.10)], PASS)
    assert track.speed_mps == pytest.approx(0.10 * 110_540 / 7200, rel=1e-3)


def test_a_single_fix_track_is_sparse_with_unknown_speed():
    (track,) = hourly_tracks([fix(hour=14)], PASS)
    assert track.sparse
    assert track.speed_mps is None
    assert not track.stationary


def test_two_fixes_in_one_cell_are_not_called_stationary_for_three_hours():
    (track,) = hourly_tracks([fix(hour=13), fix(hour=14)], PASS)
    assert not track.stationary
    assert track.speed_mps == 0.0


# ---------------------------------------------------------------------------
# Fetching presence for a pass (no network: the client is faked)
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, **values) -> None:
        self._values = values

    def model_dump(self) -> dict:
        return dict(self._values)


class _Result:
    def __init__(self, rows) -> None:
        self._rows = rows

    def data(self) -> list:
        return self._rows


class _FakeFourWings:
    def __init__(self, failures: int = 0) -> None:
        self.calls: list[dict] = []
        self.failures = failures

    async def create_ais_presence_report(self, **kwargs) -> _Result:
        self.calls.append(kwargs)
        if self.failures:
            self.failures -= 1
            raise _FlakyError("connection dropped")
        return _Result([_Row(**row(vessel_id="v9", date="2026-09-25 13:00"))])


class _FakeClient:
    def __init__(self, failures: int = 0) -> None:
        self.fourwings = _FakeFourWings(failures)


class _FlakyError(Exception):
    pass


BOX = BBox(56.43, 25.13, 56.63, 25.31)


async def test_fetch_asks_for_hourly_vessel_presence_over_the_box_for_the_pass_window():
    client = _FakeClient()
    fixes = await fetch_presence(client, BOX, PASS)
    (call,) = client.fourwings.calls
    assert (call["spatial_resolution"], call["temporal_resolution"], call["group_by"]) == (
        "HIGH",
        "HOURLY",
        "VESSEL_ID",
    )
    assert call["geojson"]["type"] == "Polygon"
    assert call["geojson"]["coordinates"][0][0] == [BOX.min_lon, BOX.min_lat]
    assert [f.vessel_id for f in fixes] == ["v9"]


async def test_fetch_renders_the_window_in_the_hour_form_4wings_accepts():
    client = _FakeClient()
    await fetch_presence(client, BOX, PASS)
    (call,) = client.fourwings.calls
    # gfwapiclient takes date objects and calls isoformat(); 4Wings rejects
    # '2026-09-25T13:00:00' but honours '.000Z' hour bounds (measured 2026-10-04).
    assert isinstance(call["start_date"], dt.date)
    assert call["start_date"].isoformat() == "2026-09-25T13:00:00.000Z"
    assert call["end_date"].isoformat() == "2026-09-25T16:00:00.000Z"


async def test_fetch_retries_transient_failures():
    client = _FakeClient(failures=2)

    async def no_sleep(_seconds: float) -> None:
        return None

    fixes = await fetch_presence(
        client, BOX, PASS, transient=lambda e: isinstance(e, _FlakyError), sleep=no_sleep
    )
    assert len(client.fourwings.calls) == 3
    assert len(fixes) == 1


def test_the_report_dataset_version_survives_parsing():
    (parsed,) = parse_presence_rows([row(report_dataset="public-global-presence:v4.0")])
    assert parsed.report_dataset == "public-global-presence:v4.0"


def test_tracks_know_the_pass_hour_and_its_fraction():
    (track,) = hourly_tracks([fix(hour=14)], PASS)
    assert track.pass_hour == dt.datetime(2026, 9, 25, 14, tzinfo=dt.UTC)
    assert track.pass_fraction == pytest.approx((15 * 60 + 51) / 3600)
