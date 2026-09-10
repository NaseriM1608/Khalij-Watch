"""Drain the ingest queue into batched database writes.

Kept separate from the socket client so that a slow or failing database can
never stall the reader. The two communicate only through a bounded queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from core.ais.aisstream import (
    POSITION_TYPES,
    STATIC_TYPES,
    AisParseError,
    parse_position,
    parse_static,
)
from core.ais.messages import AisPositionReport, VesselStatic
from core.ais.plausibility import PlausibilityAnalyzer, ScoredPosition
from core.db.repository import PositionRepository, VesselRepository
from core.db.session import session_scope
from services.ais_ingest.client import IngestStats

logger = logging.getLogger(__name__)

# Write when either threshold is reached, so a quiet feed still lands promptly.
DEFAULT_BATCH_SIZE = 500
DEFAULT_FLUSH_INTERVAL_S = 2.0


class BatchAccumulator:
    """Collects parsed messages until a batch is worth writing."""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._batch_size = batch_size
        self.positions: list[ScoredPosition] = []
        self.statics: list[VesselStatic] = []

    def add_position(self, report: ScoredPosition | AisPositionReport) -> None:
        self.positions.append(report)

    def add_static(self, static: VesselStatic) -> None:
        self.statics.append(static)

    @property
    def pending(self) -> int:
        return len(self.positions) + len(self.statics)

    @property
    def is_full(self) -> bool:
        return self.pending >= self._batch_size

    def take(self) -> tuple[list[ScoredPosition], list[VesselStatic]]:
        positions, statics = self.positions, self.statics
        self.positions, self.statics = [], []
        return positions, statics


def parse_envelope(
    source: str, envelope: dict[str, Any], *, now: dt.datetime | None = None
) -> AisPositionReport | VesselStatic | None:
    """Turn one envelope into a domain object, or ``None`` if unusable."""
    message_type = envelope.get("MessageType")
    try:
        if message_type in POSITION_TYPES:
            return parse_position(envelope, source=source, now=now)
        if message_type in STATIC_TYPES:
            return parse_static(envelope, source=source)
    except AisParseError:
        return None
    except (ValueError, TypeError, KeyError):
        # Malformed input is routine on this feed and must never crash ingest.
        return None
    return None


def _mmsi_of(entry: ScoredPosition | AisPositionReport) -> int:
    return entry.report.mmsi if isinstance(entry, ScoredPosition) else entry.mmsi


def write_batch(
    session: Session,
    positions: list[ScoredPosition] | list[AisPositionReport],
    statics: list[VesselStatic],
    *,
    now: dt.datetime | None = None,
) -> int:
    """Persist one batch. Returns the number of positions written."""
    now = now or dt.datetime.now(dt.UTC)
    vessels = VesselRepository(session)

    # Vessel rows must exist before positions reference them, and a position
    # routinely arrives before any static message for that MMSI.
    mmsis = {_mmsi_of(p) for p in positions} | {s.mmsi for s in statics}
    vessels.ensure_exists(mmsis, seen_at=now)
    session.flush()

    for static in statics:
        vessels.upsert_static(static, seen_at=now)

    written = PositionRepository(session).add_batch(positions) if positions else 0
    return written


class IngestConsumer:
    """Consumes parsed envelopes and writes them in batches."""

    def __init__(
        self,
        queue: asyncio.Queue[tuple[str, dict[str, Any]]],
        session_factory: sessionmaker[Session],
        *,
        stats: IngestStats | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        analyzer: PlausibilityAnalyzer | None = None,
    ) -> None:
        self._queue = queue
        self._session_factory = session_factory
        self._stats = stats or IngestStats()
        self._batch = BatchAccumulator(batch_size)
        self._flush_interval = flush_interval_s
        self._stop = asyncio.Event()
        # Scoring happens on the consumer side so the socket reader stays free.
        self._analyzer = analyzer or PlausibilityAnalyzer()

    @property
    def stats(self) -> IngestStats:
        return self._stats

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Consume until stopped, flushing on size or on a timer."""
        while not self._stop.is_set():
            try:
                source, envelope = await asyncio.wait_for(
                    self._queue.get(), timeout=self._flush_interval
                )
            except TimeoutError:
                self._flush()
                continue

            self._ingest(source, envelope)
            if self._batch.is_full:
                self._flush()

        self._flush()  # do not lose a partial batch on shutdown

    def _ingest(self, source: str, envelope: dict[str, Any]) -> None:
        parsed = parse_envelope(source, envelope)
        if parsed is None:
            self._stats.rejected += 1
            return
        self._stats.parsed += 1
        self._stats.sources_seen.add(source)
        if isinstance(parsed, AisPositionReport):
            verdict = self._analyzer.evaluate(parsed)
            if verdict.is_suspect:
                self._stats.suspect += 1
            self._batch.add_position(ScoredPosition(parsed, verdict))
        else:
            self._batch.add_static(parsed)

    def _flush(self) -> None:
        if not self._batch.pending:
            return
        positions, statics = self._batch.take()
        try:
            with session_scope(self._session_factory) as session:
                write_batch(session, positions, statics)
        except Exception as exc:
            self._stats.last_error = f"write failed: {type(exc).__name__}: {exc}"
            logger.exception("failed to write batch of %d", len(positions) + len(statics))


async def drain_queue(
    queue: asyncio.Queue[tuple[str, dict[str, Any]]],
    session_factory: sessionmaker[Session],
    *,
    stats: IngestStats | None = None,
) -> int:
    """Write everything currently queued, then return. Used by tests and CLI."""
    consumer = IngestConsumer(queue, session_factory, stats=stats)
    while not queue.empty():
        source, envelope = queue.get_nowait()
        consumer._ingest(source, envelope)
    consumer._flush()
    return consumer.stats.parsed


async def run_until_stopped(task: asyncio.Task[Any]) -> None:
    """Await a task, swallowing cancellation during shutdown."""
    with contextlib.suppress(asyncio.CancelledError):
        await task
