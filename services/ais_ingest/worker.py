"""Runnable AIS ingest worker.

Run:  uv run python -m services.ais_ingest.worker

Requires AISSTREAM_API_KEY (or AISCAST_TOKEN) in .env. Without one it exits
with an explanation rather than failing obscurely.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from core.ais.aisstream import build_subscription
from core.config import Settings, load_settings
from core.db.session import create_all, create_db_engine, make_session_factory
from services.ais_ingest.client import (
    DEFAULT_QUEUE_SIZE,
    AisFeedClient,
    FeedConfig,
    IngestStats,
    WebSocketLike,
)
from services.ais_ingest.consumer import IngestConsumer

logger = logging.getLogger("ais_ingest")

STATS_INTERVAL_S = 30.0


async def websocket_connector(url: str) -> WebSocketLike:
    """Open a real websocket. Imported lazily so tests need no dependency."""
    import websockets

    # A ping keeps NAT and proxy paths from silently dropping an idle socket.
    return await websockets.connect(url, ping_interval=20, ping_timeout=20)


def build_feeds(settings: Settings) -> list[FeedConfig]:
    """Build feed configs for whichever credentials are present."""
    bboxes = settings.gulf_bbox.as_aisstream()
    feeds: list[FeedConfig] = []

    if settings.aisstream_key:
        feeds.append(
            FeedConfig(
                name="aisstream",
                url=settings.aisstream_url,
                subscription=build_subscription(settings.aisstream_key, bboxes),
                source="aisstream",
            )
        )
    if settings.aiscast_token:
        # aiscast speaks the same protocol, so the same frame works.
        feeds.append(
            FeedConfig(
                name="aiscast",
                url=settings.aiscast_url,
                subscription=build_subscription(settings.aiscast_token, bboxes),
                source="aiscast",
            )
        )
    return feeds


async def report_stats(stats: IngestStats, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=STATS_INTERVAL_S)
        if stop.is_set():
            return
        logger.info("ingest stats: %s", stats.as_dict())


async def run(settings: Settings) -> int:
    feeds = build_feeds(settings)
    if not feeds:
        logger.error(
            "No AIS credentials configured. Set AISSTREAM_API_KEY (free at "
            "https://aisstream.io/authenticate) or AISCAST_TOKEN in .env"
        )
        return 1

    engine = create_db_engine(settings.database_url)
    create_all(engine)
    session_factory = make_session_factory(engine)

    queue: asyncio.Queue = asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE)
    stats = IngestStats()
    clients = [AisFeedClient(f, queue, websocket_connector, stats=stats) for f in feeds]
    consumer = IngestConsumer(queue, session_factory, stats=stats)

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    logger.info(
        "starting ingest: feeds=%s bbox=%s",
        [f.name for f in feeds],
        settings.gulf_bbox,
    )
    tasks = [asyncio.create_task(c.run()) for c in clients]
    tasks.append(asyncio.create_task(consumer.run()))
    tasks.append(asyncio.create_task(report_stats(stats, stop)))

    await stop.wait()
    logger.info("shutting down; final stats: %s", stats.as_dict())

    for client in clients:
        client.stop()
    consumer.stop()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return 0


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            # Windows does not support add_signal_handler for all signals.
            loop.add_signal_handler(sig, stop.set)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    settings = load_settings()
    try:
        return asyncio.run(run(settings))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
