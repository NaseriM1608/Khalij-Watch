"""WebSocket client for aisstream.io and the aiscast fallback.

Two server behaviours drive the whole design:

1. **The subscription must arrive within 3 seconds of connecting**, or the
   server closes the socket.
2. **The server drops messages if the client consumes too slowly.** So the
   socket reader must never block on the database. It pushes into a bounded
   queue and, if that queue is full, drops locally and counts it. A visible
   local drop is far better than silent server-side truncation.

The transport is injectable so the reconnect and backpressure logic can be
tested without a live server or network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# The server closes the socket if the subscription is late. Send it first,
# well inside the window.
SUBSCRIBE_DEADLINE_S = 3.0

# Reconnect backoff. Jitter avoids a thundering herd if the service restarts
# at the same moment as other clients.
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
BACKOFF_FACTOR = 2.0

# Bounded so a database stall cannot grow memory without limit.
DEFAULT_QUEUE_SIZE = 10_000


class WebSocketLike(Protocol):
    """The subset of a websockets connection this client uses."""

    async def send(self, message: str) -> None: ...
    def __aiter__(self) -> AsyncIterator[str]: ...


Connector = Callable[[str], Awaitable[WebSocketLike]]


@dataclass
class IngestStats:
    """Counters for observability. A silent feed must be diagnosable."""

    connects: int = 0
    disconnects: int = 0
    received: int = 0
    parsed: int = 0
    rejected: int = 0
    dropped_queue_full: int = 0
    suspect: int = 0
    # Server control messages, kept apart from AIS traffic so that a quiet
    # feed is not mistaken for a parsing fault, and a refusal is not lost.
    confirmations: int = 0
    server_errors: int = 0
    last_error: str | None = None
    sources_seen: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "connects": self.connects,
            "disconnects": self.disconnects,
            "received": self.received,
            "parsed": self.parsed,
            "rejected": self.rejected,
            "confirmations": self.confirmations,
            "server_errors": self.server_errors,
            "dropped_queue_full": self.dropped_queue_full,
            "suspect": self.suspect,
            "last_error": self.last_error,
            "sources_seen": sorted(self.sources_seen),
        }


def next_backoff(attempt: int, *, rng: random.Random | None = None) -> float:
    """Exponential backoff with full jitter, capped at :data:`MAX_BACKOFF_S`."""
    rng = rng or random
    ceiling = min(INITIAL_BACKOFF_S * (BACKOFF_FACTOR**attempt), MAX_BACKOFF_S)
    return rng.uniform(0.0, ceiling)


@dataclass(frozen=True)
class FeedConfig:
    """One upstream feed. ``source`` is stored on every row for attribution."""

    name: str
    url: str
    subscription: dict[str, Any]
    source: str

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError(f"feed {self.name!r} has no URL")
        if not self.source:
            raise ValueError(f"feed {self.name!r} has no source label")


class AisFeedClient:
    """Maintains a subscription to one feed, reconnecting as needed."""

    def __init__(
        self,
        config: FeedConfig,
        queue: asyncio.Queue[tuple[str, dict[str, Any]]],
        connector: Connector,
        *,
        stats: IngestStats | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._connector = connector
        self._stats = stats or IngestStats()
        self._rng = rng
        self._stop = asyncio.Event()

    @property
    def stats(self) -> IngestStats:
        return self._stats

    def stop(self) -> None:
        self._stop.set()

    async def run(self, *, max_attempts: int | None = None) -> None:
        """Connect and stream until stopped.

        ``max_attempts`` bounds reconnection, which tests rely on; in
        production it is left unbounded.
        """
        attempt = 0
        while not self._stop.is_set():
            if max_attempts is not None and attempt >= max_attempts:
                return
            try:
                await self._session()
                # A clean close still counts as a failure for backoff purposes:
                # the feed is meant to be continuous.
                attempt += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                self._stats.last_error = f"{type(exc).__name__}: {exc}"
                self._stats.disconnects += 1
                logger.warning("feed %s failed (attempt %d): %s", self._config.name, attempt, exc)

            if self._stop.is_set():
                return
            delay = next_backoff(attempt, rng=self._rng)
            logger.info("feed %s reconnecting in %.1fs", self._config.name, delay)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=delay)

    async def _session(self) -> None:
        websocket = await self._connector(self._config.url)
        self._stats.connects += 1
        logger.info("feed %s connected", self._config.name)

        # Subscribe immediately; the server enforces a 3 second deadline.
        await asyncio.wait_for(
            websocket.send(json.dumps(self._config.subscription)),
            timeout=SUBSCRIBE_DEADLINE_S,
        )

        async for raw in websocket:
            if self._stop.is_set():
                return
            self._stats.received += 1
            envelope = _decode(raw)
            if envelope is None:
                self._stats.rejected += 1
                continue
            if self._handle_control(envelope):
                continue
            self._offer(envelope)

    def _handle_control(self, envelope: dict[str, Any]) -> bool:
        """Consume a server control message. Returns True if ``envelope`` was one."""
        if envelope.get("MessageType") == "SubscriptionConfirmation":
            self._stats.confirmations += 1
            logger.info("feed %s subscription confirmed", self._config.name)
            return True
        if "error" in envelope:
            self._stats.server_errors += 1
            self._stats.last_error = f"server: {envelope['error']}"
            logger.error("feed %s reported an error: %s", self._config.name, envelope["error"])
            return True
        return False

    def _offer(self, envelope: dict[str, Any]) -> None:
        """Hand a message to the consumer without ever blocking the reader."""
        try:
            self._queue.put_nowait((self._config.source, envelope))
        except asyncio.QueueFull:
            # Dropping here is deliberate. Blocking would make the server drop
            # messages for us, invisibly; this way the loss is counted.
            self._stats.dropped_queue_full += 1
            if self._stats.dropped_queue_full % 1000 == 1:
                logger.warning(
                    "ingest queue full; dropped %d messages so far",
                    self._stats.dropped_queue_full,
                )


def _decode(raw: object) -> dict[str, Any] | None:
    if isinstance(raw, bytes | bytearray):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
