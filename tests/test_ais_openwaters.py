"""Native OpenWaters position events retain time and upstream provenance."""

from __future__ import annotations

import asyncio
import datetime as dt
import json

import httpx
import pytest

from core.ais.messages import AisPositionReport, VesselStatic
from services.ais_ingest.client import IngestStats
from services.ais_ingest.consumer import parse_envelope

NOW = dt.datetime(2026, 10, 29, 12, tzinfo=dt.UTC)


def event(**overrides):
    value = {
        "type": "event",
        "id": "example",
        "time": "2026-10-29T11:59:00Z",
        "source": "aishub",
        "mmsi": 422123456,
        "msg_type": "PositionReport",
        "lat": 26.5,
        "lon": 56.5,
        "synthesized": True,
        "message": {
            "MessageID": 1,
            "UserID": 422123456,
            "Latitude": 26.5,
            "Longitude": 56.5,
            "Sog": 12.0,
            "Cog": 90.0,
            "Valid": True,
        },
    }
    value.update(overrides)
    return value


def test_native_position_retains_its_event_time_and_provider_reencoding():
    parsed = parse_envelope("openwaters", event(), now=NOW)
    assert isinstance(parsed, AisPositionReport)
    assert parsed.ts == NOW - dt.timedelta(seconds=60)
    assert parsed.received_at == NOW
    assert parsed.timestamp_basis == "provider"
    assert parsed.source == "openwaters"
    assert parsed.provider_source == "aishub"
    assert parsed.provider_reencoded is True
    assert (parsed.sog_kn, parsed.cog_deg) == (12.0, 90.0)


def test_static_event_with_top_level_coordinates_never_becomes_a_position():
    parsed = parse_envelope(
        "openwaters",
        event(msg_type="ShipStaticData", message={"UserID": 422123456, "Name": "TEST", "Type": 70}),
        now=NOW,
    )
    assert isinstance(parsed, VesselStatic)


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "welcome"},
        {"message": {"UserID": 422123456}},
        {"message": {"UserID": 422123456, "Latitude": 26.5, "Longitude": 56.5, "Valid": False}},
        {"msg_type": "Unknown"},
    ],
)
def test_controls_and_invalid_or_nonposition_bodies_are_not_fresh_fixes(overrides):
    assert parse_envelope("openwaters", event(**overrides), now=NOW) is None


def test_bad_event_clock_is_retained_as_unknown_time_not_a_provider_observation():
    parsed = parse_envelope("openwaters", event(time="bad"), now=NOW)
    assert parsed.timestamp_basis == "receipt_fallback"
    assert parsed.ts == NOW


def test_sse_client_handles_comments_multiline_json_and_bounded_queue():
    from services.ais_ingest.openwaters import OpenWatersClient

    async def lines(_url):
        yield ": heartbeat"
        yield ""
        yield 'data: {"type":"welcome"}'
        yield ""
        for _ in range(3):
            payload = json.dumps(event())
            yield "event: message"
            yield "data: " + payload[: payload.index(', "source"')]
            yield "data: " + payload[payload.index(', "source"') :]
            yield ""

    async def run():
        queue = asyncio.Queue(maxsize=1)
        stats = IngestStats()
        client = OpenWatersClient(
            "https://example.invalid/stream?bbox=1,2,3,4", queue, lines=lines, stats=stats
        )
        await client.run(max_attempts=1)
        assert stats.received == 3
        assert stats.dropped_queue_full == 2
        source, queued = queue.get_nowait()
        assert source == "openwaters" and queued["source"] == "aishub"
        assert queued["_received_at"] is not None

    asyncio.run(run())


def test_sse_transport_failure_reconnects_without_rewriting_provider_event(monkeypatch):
    from services.ais_ingest import openwaters

    monkeypatch.setattr(openwaters, "next_backoff", lambda _attempt: 0)
    attempts = []

    async def lines(_url):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("temporary disconnect")
        yield "data: not-json"
        yield ""
        yield "data: " + json.dumps(event())
        yield ""

    async def run():
        queue = asyncio.Queue()
        stats = IngestStats()
        client = openwaters.OpenWatersClient(
            "https://example.invalid", queue, lines=lines, stats=stats
        )
        await client.run(max_attempts=2)
        assert stats.connects == 2
        assert stats.rejected == 1
        assert queue.get_nowait()[1]["time"] == "2026-10-29T11:59:00Z"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("retry_after", "expected"), [("120", 120), ("2", 30), ("bad", 30), ("nan", 30)]
)
def test_sse_rate_limit_honors_server_delay_with_minimum_thirty_seconds(
    monkeypatch, caplog, retry_after, expected
):
    from services.ais_ingest import openwaters

    delays = []
    monkeypatch.setattr(openwaters, "next_backoff", lambda _attempt: 1)

    async def lines(_url):
        yield ": connected"
        response = httpx.Response(
            429,
            headers={"Retry-After": retry_after},
            content=b"PRIVATE_BODY",
            request=httpx.Request("GET", "https://example.invalid"),
        )
        response.raise_for_status()

    async def run():
        stats = IngestStats()
        client = openwaters.OpenWatersClient(
            "https://example.invalid", asyncio.Queue(), lines=lines, stats=stats
        )

        async def record_wait(awaitable, *, timeout):
            awaitable.close()
            delays.append(timeout)
            client.stop()

        monkeypatch.setattr(openwaters.asyncio, "wait_for", record_wait)
        await client.run()
        assert delays == [expected]
        assert "429" in stats.last_error
        assert "PRIVATE_BODY" not in caplog.text

    asyncio.run(run())
