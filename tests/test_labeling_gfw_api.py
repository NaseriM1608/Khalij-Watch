"""Tests for fetching GFW SAR presence cells (no network: the client is faked)."""

from __future__ import annotations

import datetime as dt

import pytest

from core.db.spatial import BBox
from core.labeling.gfw_api import fetch_sar_cells, month_ranges, with_retries

BOX = BBox(56.43, 25.13, 56.63, 25.31)


class Row:
    def __init__(self, **values) -> None:
        self._values = values

    def model_dump(self) -> dict:
        return dict(self._values)


class Result:
    def __init__(self, rows) -> None:
        self._rows = rows

    def data(self) -> list[Row]:
        return self._rows


class FakeFourWings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create_sar_presence_report(self, **kwargs) -> Result:
        self.calls.append(kwargs)
        if kwargs.get("filters") == ["matched='true'"]:
            return Result(
                [
                    Row(
                        date="2025-09-17 14:00",
                        detections=1,
                        lat=25.25,
                        lon=56.48,
                        mmsi="1",
                        vessel_id="v1",
                    )
                ]
            )
        return Result([Row(date="2025-09-17 14:00", detections=2, lat=25.20, lon=56.50)])


class FakeClient:
    def __init__(self) -> None:
        self.fourwings = FakeFourWings()


class FlakyError(Exception):
    pass


async def no_sleep(_seconds: float) -> None:
    return None


# --------------------------------------------------------------------------
# Date chunking
# --------------------------------------------------------------------------


def test_a_single_month_is_one_range():
    assert month_ranges(dt.date(2025, 9, 1), dt.date(2025, 10, 1)) == [
        (dt.date(2025, 9, 1), dt.date(2025, 10, 1))
    ]


def test_ranges_are_cut_at_month_boundaries():
    assert month_ranges(dt.date(2025, 8, 15), dt.date(2025, 10, 10)) == [
        (dt.date(2025, 8, 15), dt.date(2025, 9, 1)),
        (dt.date(2025, 9, 1), dt.date(2025, 10, 1)),
        (dt.date(2025, 10, 1), dt.date(2025, 10, 10)),
    ]


def test_an_empty_range_is_refused():
    with pytest.raises(ValueError, match="before"):
        month_ranges(dt.date(2025, 9, 1), dt.date(2025, 9, 1))


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------


async def test_transient_failures_are_retried():
    """This host drops connections intermittently; the first GFW calls all failed."""
    attempts = []

    async def call():
        attempts.append(1)
        if len(attempts) < 3:
            raise FlakyError("connection error")
        return "ok"

    result = await with_retries(
        call, is_transient=lambda e: isinstance(e, FlakyError), sleep=no_sleep
    )

    assert result == "ok"
    assert len(attempts) == 3


async def test_other_errors_are_not_retried():
    attempts = []

    async def call():
        attempts.append(1)
        raise ValueError("422 unprocessable")

    with pytest.raises(ValueError):
        await with_retries(call, is_transient=lambda e: isinstance(e, FlakyError), sleep=no_sleep)
    assert len(attempts) == 1


async def test_retries_give_up_eventually():
    async def call():
        raise FlakyError("still down")

    with pytest.raises(FlakyError):
        await with_retries(
            call, is_transient=lambda e: isinstance(e, FlakyError), attempts=3, sleep=no_sleep
        )


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


async def test_matched_and_unmatched_cells_are_both_fetched_per_month():
    client = FakeClient()

    cells = await fetch_sar_cells(client, BOX, dt.date(2025, 9, 1), dt.date(2025, 10, 1))

    assert sorted(c.matched for c in cells) == [False, True]
    matched = next(c for c in cells if c.matched)
    assert matched.mmsi == 1 and matched.vessel_id == "v1"
    assert next(c for c in cells if not c.matched).detections == 2


async def test_requests_use_the_parameters_verified_against_the_live_api():
    client = FakeClient()

    await fetch_sar_cells(client, BOX, dt.date(2025, 8, 20), dt.date(2025, 10, 1))

    calls = client.fourwings.calls
    assert len(calls) == 4  # two months, matched and unmatched each
    for call in calls:
        assert call["spatial_resolution"] == "HIGH"
        assert call["temporal_resolution"] == "HOURLY"
        assert call["geojson"]["type"] == "Polygon"
    matched_calls = [c for c in calls if c["filters"] == ["matched='true'"]]
    assert [c["group_by"] for c in matched_calls] == ["VESSEL_ID", "VESSEL_ID"]
    unmatched_calls = [c for c in calls if c["filters"] == ["matched='false'"]]
    assert all(c.get("group_by") is None for c in unmatched_calls)
    assert calls[0]["start_date"] == "2025-08-20"
    assert calls[-1]["end_date"] == "2025-10-01"
