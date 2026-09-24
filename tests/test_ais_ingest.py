"""Tests for the AIS ingest client and consumer.

The transport is faked, so reconnection and backpressure are tested without a
network. The behaviours that matter are the ones the server imposes on us:
subscribe within 3 seconds, and never consume too slowly.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import random

import pytest

from core.ais.messages import AisPositionReport, VesselStatic
from core.db.repository import PositionRepository, VesselRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from services.ais_ingest.client import (
    MAX_BACKOFF_S,
    AisFeedClient,
    FeedConfig,
    IngestStats,
    next_backoff,
)
from services.ais_ingest.consumer import (
    BatchAccumulator,
    IngestConsumer,
    parse_envelope,
    write_batch,
)

MMSI = 422123456
NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)

pytestmark = pytest.mark.asyncio


def position_envelope(mmsi=MMSI, lat=26.5, lon=56.5):
    return {
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": mmsi, "time_utc": "2026-09-10 11:59:30.0 +0000 UTC"},
        "Message": {
            "PositionReport": {
                "UserID": mmsi,
                "Latitude": lat,
                "Longitude": lon,
                "Sog": 12.0,
                "Cog": 90.0,
                "TrueHeading": 90,
                "MessageID": 1,
            }
        },
    }


def static_envelope(mmsi=MMSI, name="EVER GIVEN"):
    return {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": mmsi},
        "Message": {"ShipStaticData": {"UserID": mmsi, "Name": name, "Type": 70}},
    }


class FakeSocket:
    """A scripted websocket. Records what was sent, replays what was queued."""

    def __init__(self, messages: list[str], *, fail_after: int | None = None) -> None:
        self._messages = messages
        self._fail_after = fail_after
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def __aiter__(self):
        for index, message in enumerate(self._messages):
            if self._fail_after is not None and index >= self._fail_after:
                raise ConnectionError("upstream closed")
            yield message


def make_config(**overrides) -> FeedConfig:
    defaults = {
        "name": "test",
        "url": "wss://example.invalid/stream",
        "subscription": {"APIKey": "K", "BoundingBoxes": [[[24.5, 54.0], [27.5, 57.5]]]},
        "source": "aisstream",
    }
    defaults.update(overrides)
    return FeedConfig(**defaults)


@pytest.fixture
def session_factory():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    return make_session_factory(engine)


# --------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------


async def test_backoff_grows_and_is_capped():
    rng = random.Random(0)
    # Full jitter means the value is bounded by the ceiling, not equal to it.
    assert next_backoff(0, rng=rng) <= 1.0
    assert next_backoff(3, rng=rng) <= 8.0
    assert next_backoff(50, rng=rng) <= MAX_BACKOFF_S


async def test_backoff_is_never_negative():
    rng = random.Random(1)
    assert all(next_backoff(i, rng=rng) >= 0 for i in range(20))


# --------------------------------------------------------------------------
# Client behaviour
# --------------------------------------------------------------------------


async def test_subscription_is_sent_before_any_message_is_read():
    """The server closes the socket if the subscription is late."""
    socket = FakeSocket([json.dumps(position_envelope())])
    queue: asyncio.Queue = asyncio.Queue()
    config = make_config()
    client = AisFeedClient(config, queue, lambda _url: _ready(socket))

    await client.run(max_attempts=1)

    assert socket.sent, "no subscription was sent"
    assert json.loads(socket.sent[0]) == config.subscription


async def test_messages_are_queued_with_their_source():
    """Every row must carry its source for licence attribution."""
    socket = FakeSocket([json.dumps(position_envelope())])
    queue: asyncio.Queue = asyncio.Queue()
    client = AisFeedClient(make_config(source="aiscast"), queue, lambda _u: _ready(socket))

    await client.run(max_attempts=1)

    source, envelope = queue.get_nowait()
    assert source == "aiscast"
    assert envelope["MessageType"] == "PositionReport"


async def test_malformed_json_is_counted_not_fatal():
    socket = FakeSocket(["{not json", json.dumps(position_envelope())])
    queue: asyncio.Queue = asyncio.Queue()
    stats = IngestStats()
    client = AisFeedClient(make_config(), queue, lambda _u: _ready(socket), stats=stats)

    await client.run(max_attempts=1)

    assert stats.rejected == 1
    assert queue.qsize() == 1


async def test_full_queue_drops_locally_rather_than_blocking_the_reader():
    """Blocking would make the SERVER drop messages for us, invisibly."""
    messages = [json.dumps(position_envelope()) for _ in range(5)]
    socket = FakeSocket(messages)
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    stats = IngestStats()
    client = AisFeedClient(make_config(), queue, lambda _u: _ready(socket), stats=stats)

    await asyncio.wait_for(client.run(max_attempts=1), timeout=5)

    assert queue.qsize() == 2
    assert stats.dropped_queue_full == 3
    assert stats.received == 5


CONFIRMATION = json.dumps(
    {"Message": {"CompressionEnabled": True}, "MessageType": "SubscriptionConfirmation"}
)


async def test_a_subscription_confirmation_is_not_counted_as_rejected():
    """Regression: the live feed's first message is a confirmation, which the
    stats reported as a rejected message, so a silent feed looked like a
    parsing fault."""
    socket = FakeSocket([CONFIRMATION, json.dumps(position_envelope())])
    queue: asyncio.Queue = asyncio.Queue()
    stats = IngestStats()
    client = AisFeedClient(make_config(), queue, lambda _u: _ready(socket), stats=stats)

    await client.run(max_attempts=1)

    assert stats.confirmations == 1
    assert stats.rejected == 0
    assert queue.qsize() == 1  # only the position reaches the consumer


async def test_a_server_error_is_surfaced_not_swallowed():
    """A bad key or subscription is explained once, then the feed goes quiet."""
    socket = FakeSocket([json.dumps({"error": "Api Key Is Not Valid"})])
    queue: asyncio.Queue = asyncio.Queue()
    stats = IngestStats()
    client = AisFeedClient(make_config(), queue, lambda _u: _ready(socket), stats=stats)

    await client.run(max_attempts=1)

    assert stats.server_errors == 1
    assert "Api Key Is Not Valid" in (stats.last_error or "")
    assert queue.qsize() == 0


def test_stats_report_confirmations_and_server_errors():
    stats = IngestStats(confirmations=2, server_errors=1)

    assert stats.as_dict()["confirmations"] == 2
    assert stats.as_dict()["server_errors"] == 1


async def test_connection_failure_is_recorded_and_retried():
    attempts = []

    async def failing_connector(url: str):
        attempts.append(url)
        raise ConnectionError("refused")

    queue: asyncio.Queue = asyncio.Queue()
    stats = IngestStats()
    # Zero backoff keeps the test fast.
    rng = random.Random()
    rng.uniform = lambda _a, _b: 0.0  # type: ignore[method-assign]
    client = AisFeedClient(make_config(), queue, failing_connector, stats=stats, rng=rng)

    await asyncio.wait_for(client.run(max_attempts=3), timeout=5)

    assert len(attempts) == 3
    assert stats.disconnects == 3
    assert "ConnectionError" in (stats.last_error or "")


async def test_mid_stream_failure_still_keeps_earlier_messages():
    socket = FakeSocket([json.dumps(position_envelope())] * 4, fail_after=2)
    queue: asyncio.Queue = asyncio.Queue()
    stats = IngestStats()
    rng = random.Random()
    rng.uniform = lambda _a, _b: 0.0  # type: ignore[method-assign]
    client = AisFeedClient(make_config(), queue, lambda _u: _ready(socket), stats=stats, rng=rng)

    await asyncio.wait_for(client.run(max_attempts=1), timeout=5)

    assert queue.qsize() == 2


async def test_stop_ends_the_run_loop():
    socket = FakeSocket([])
    queue: asyncio.Queue = asyncio.Queue()
    client = AisFeedClient(make_config(), queue, lambda _u: _ready(socket))
    client.stop()

    await asyncio.wait_for(client.run(), timeout=5)


async def test_feed_config_requires_a_source_label():
    with pytest.raises(ValueError, match="source"):
        make_config(source="")


# --------------------------------------------------------------------------
# Parsing and batching
# --------------------------------------------------------------------------


async def test_parse_envelope_routes_by_message_type():
    assert isinstance(parse_envelope("t", position_envelope(), now=NOW), AisPositionReport)
    assert isinstance(parse_envelope("t", static_envelope()), VesselStatic)


async def test_parse_envelope_returns_none_for_junk():
    assert parse_envelope("t", {}, now=NOW) is None
    assert parse_envelope("t", {"MessageType": "Unknown"}, now=NOW) is None
    # Invalid MMSI must be rejected, not raised.
    assert parse_envelope("t", position_envelope(mmsi=0), now=NOW) is None


async def test_batch_accumulator_reports_fullness_across_both_kinds():
    batch = BatchAccumulator(batch_size=3)
    batch.add_position(AisPositionReport(mmsi=MMSI, ts=NOW, lon=56.5, lat=26.5, source="t"))
    batch.add_static(VesselStatic(mmsi=MMSI))
    assert not batch.is_full

    batch.add_static(VesselStatic(mmsi=MMSI))
    assert batch.is_full


async def test_take_empties_the_accumulator():
    batch = BatchAccumulator()
    batch.add_static(VesselStatic(mmsi=MMSI))

    _positions, statics = batch.take()

    assert len(statics) == 1
    assert batch.pending == 0


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


async def test_write_batch_creates_vessels_before_positions(session_factory):
    """A position routinely arrives before any static message for that MMSI."""
    report = AisPositionReport(mmsi=MMSI, ts=NOW, lon=56.5, lat=26.5, source="t")

    with session_scope(session_factory) as session:
        written = write_batch(session, [report], [], now=NOW)

    assert written == 1
    with session_scope(session_factory) as session:
        assert VesselRepository(session).get(MMSI) is not None


async def test_write_batch_applies_static_data(session_factory):
    with session_scope(session_factory) as session:
        write_batch(session, [], [VesselStatic(mmsi=MMSI, name="X", ship_type=70)], now=NOW)

    with session_scope(session_factory) as session:
        vessel = VesselRepository(session).get(MMSI)
        assert vessel.name == "X"
        assert vessel.coarse_class == "cargo_tanker"


async def test_consumer_writes_queued_messages(session_factory):
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(("aisstream", position_envelope()))
    await queue.put(("aisstream", static_envelope()))
    consumer = IngestConsumer(queue, session_factory, batch_size=2, flush_interval_s=0.05)

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.3)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5)

    with session_scope(session_factory) as session:
        assert PositionRepository(session).count() == 1
        assert VesselRepository(session).get(MMSI).name == "EVER GIVEN"


async def test_consumer_flushes_a_partial_batch_on_a_timer(session_factory):
    """A quiet feed must still land its rows."""
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(("aisstream", position_envelope()))
    consumer = IngestConsumer(queue, session_factory, batch_size=500, flush_interval_s=0.05)

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.3)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5)

    with session_scope(session_factory) as session:
        assert PositionRepository(session).count() == 1


async def test_consumer_counts_rejects_without_crashing(session_factory):
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(("aisstream", {"MessageType": "Nonsense"}))
    consumer = IngestConsumer(queue, session_factory, flush_interval_s=0.05)

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.2)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5)

    assert consumer.stats.rejected == 1


async def test_consumer_records_sources_for_attribution(session_factory):
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(("aiscast", position_envelope()))
    consumer = IngestConsumer(queue, session_factory, batch_size=1, flush_interval_s=0.05)

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.2)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5)

    assert consumer.stats.sources_seen == {"aiscast"}


async def _ready(socket):
    return socket


async def test_spoofed_track_is_stored_with_its_flags(session_factory):
    """A fabricated position is FLAGGED AND KEPT, never dropped.

    In this AOI a falsified track is itself evidence: it identifies a vessel
    actively concealing movement. Discarding it would destroy the signal.
    """
    from sqlalchemy import text

    from core.ais.landmask import NoLandMask
    from core.ais.plausibility import PlausibilityAnalyzer

    queue: asyncio.Queue = asyncio.Queue()
    # Two fixes ~550 km apart, 30 s apart: physically impossible.
    await queue.put(("aisstream", position_envelope(lat=26.5, lon=56.5)))
    await queue.put(("aisstream", position_envelope(lat=27.5, lon=51.0)))

    consumer = IngestConsumer(
        queue,
        session_factory,
        batch_size=2,
        flush_interval_s=0.05,
        analyzer=PlausibilityAnalyzer(land_mask=NoLandMask()),
    )
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.3)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5)

    with session_scope(session_factory) as session:
        # Both positions are retained.
        assert PositionRepository(session).count() == 2
        rows = session.execute(
            text(
                "SELECT plausibility_score, anomaly_flags FROM ais_positions "
                "ORDER BY plausibility_score"
            )
        ).all()

    worst_score, worst_flags = rows[0]
    assert worst_score < 0.5
    assert "TELEPORT" in worst_flags
    assert consumer.stats.suspect == 1


async def test_ordinary_traffic_is_scored_clean(session_factory):
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(("aisstream", position_envelope(lat=26.500, lon=56.500)))
    consumer = IngestConsumer(queue, session_factory, batch_size=1, flush_interval_s=0.05)

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.25)
    consumer.stop()
    await asyncio.wait_for(task, timeout=5)

    assert consumer.stats.suspect == 0
