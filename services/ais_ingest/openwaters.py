"""Anonymous native SSE ingestion through the existing bounded AIS queue."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import math
from collections.abc import AsyncIterator, Callable

import httpx

from services.ais_ingest.client import IngestStats, next_backoff

logger = logging.getLogger(__name__)


async def http_lines(url: str) -> AsyncIterator[str]:
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=45.0)) as client,
        client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            yield line


class OpenWatersClient:
    def __init__(
        self,
        url: str,
        queue: asyncio.Queue,
        *,
        stats: IngestStats | None = None,
        lines: Callable[[str], AsyncIterator[str]] = http_lines,
    ) -> None:
        self._url, self._queue, self._lines = url, queue, lines
        self._stats = stats or IngestStats()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self, *, max_attempts: int | None = None) -> None:
        attempt = 0
        while not self._stop.is_set() and (max_attempts is None or attempt < max_attempts):
            attempt += 1
            delay = next_backoff(min(attempt, 10))
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except (httpx.HTTPError, OSError) as exc:
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                self._stats.last_error = f"openwaters: {type(exc).__name__}, status={status}"
                if status == 429:
                    try:
                        server_delay = float(exc.response.headers.get("Retry-After", ""))
                    except ValueError:
                        server_delay = 0.0
                    delay = max(30.0, server_delay if math.isfinite(server_delay) else 0.0)
                logger.warning("OpenWaters disconnected: %s, status=%s", type(exc).__name__, status)
            self._stats.disconnects += 1
            if self._stop.is_set() or (max_attempts is not None and attempt >= max_attempts):
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=delay)

    async def _session(self) -> None:
        data: list[str] = []
        self._stats.connects += 1
        async for line in self._lines(self._url):
            if self._stop.is_set():
                return
            if line == "":
                if data:
                    self._offer("\n".join(data))
                    data = []
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))

    def _offer(self, raw: str) -> None:
        try:
            event = json.loads(raw)
        except ValueError:
            self._stats.rejected += 1
            return
        if not isinstance(event, dict) or event.get("type") != "event":
            return
        self._stats.received += 1
        event["_received_at"] = dt.datetime.now(dt.UTC)
        try:
            self._queue.put_nowait(("openwaters", event))
        except asyncio.QueueFull:
            self._stats.dropped_queue_full += 1
