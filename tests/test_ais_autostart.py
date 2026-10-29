"""The application owns one ingest task and stops it on shutdown."""

import asyncio
import datetime as dt
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.config import Settings, load_settings
from services.ais_ingest import worker
from services.api import main as api


def test_default_configuration_enables_native_feed_and_app_worker(monkeypatch, tmp_path):
    monkeypatch.delenv("AIS_AUTOSTART_ENABLED")
    monkeypatch.delenv("OPENWATERS_ENABLED", raising=False)
    settings = load_settings(tmp_path / "missing.env")
    assert settings.ais_autostart_enabled is True
    assert settings.openwaters_enabled is True


@pytest.mark.parametrize("enabled", [True, False])
def test_api_starts_once_and_stops_its_worker(monkeypatch, enabled):
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setenv("AIS_AUTOSTART_ENABLED", "1" if enabled else "0")
    started, stopped = [], []

    async def ingest(_settings, *, sessions=None, stop=None, install_signals=True):
        started.append(1)
        assert install_signals is False
        assert sessions is api._state["sessions"]
        try:
            await stop.wait()
        finally:
            stopped.append(1)

    monkeypatch.setattr(worker, "run", ingest)
    with TestClient(api.app) as client:
        first = client.get("/api/health").json()
        client.get("/api/health")
        assert first["ais_worker_running"] is enabled
        assert len(started) == int(enabled)
    assert len(stopped) == int(enabled)
    assert "ais_worker" not in api._state


def test_cancelled_worker_closes_owned_tasks_and_flushes(monkeypatch):
    from core.db.repository import PositionRepository
    from core.db.session import create_all, create_db_engine, make_session_factory, session_scope

    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    closed = []

    class Feed:
        def __init__(self, _url, queue, **_kwargs):
            self.queue = queue

        def stop(self):
            pass

        async def run(self):
            await self.queue.put(
                (
                    "openwaters",
                    {
                        "type": "event",
                        "msg_type": "PositionReport",
                        "source": "aishub",
                        "time": "2026-10-30T08:00:00Z",
                        "message": {
                            "UserID": 422123456,
                            "Latitude": 26.5,
                            "Longitude": 56.5,
                            "MessageID": 1,
                            "Sog": 10,
                            "Cog": 90,
                        },
                    },
                )
            )
            try:
                await asyncio.Event().wait()
            finally:
                closed.append(True)

    monkeypatch.setattr(worker, "OpenWatersClient", Feed)
    settings = Settings(
        aisstream_key=None,
        aisstream_url="wss://example.invalid",
        aiscast_url="wss://example.invalid",
        aiscast_token=None,
        gulf_bbox=load_settings(Path("missing.env")).gulf_bbox,
        hormuz_bbox=load_settings(Path("missing.env")).hormuz_bbox,
        database_url=None,
        data_dir=Path("data"),
        openwaters_enabled=True,
    )

    async def scenario():
        task = asyncio.create_task(worker.run(settings, sessions=factory, install_signals=False))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]

    asyncio.run(scenario())
    with session_scope(factory) as session:
        assert PositionRepository(session).count() == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_app_ingests_identity_and_position_from_a_local_native_feed(monkeypatch):
    from core.db.repository import PositionRepository
    from core.db.session import session_scope

    peer_closed = asyncio.Event()
    now = dt.datetime.now(dt.UTC).isoformat()

    async def serve(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            events = [
                {
                    "msg_type": "PositionReport",
                    "message": {
                        "UserID": 422123456,
                        "Latitude": 26.5,
                        "Longitude": 56.5,
                        "MessageID": 1,
                        "Sog": 10,
                        "Cog": 90,
                    },
                },
                {
                    "msg_type": "ShipStaticData",
                    "message": {
                        "UserID": 422123456,
                        "Name": "LOCAL TEST",
                        "Type": 80,
                        "ImoNumber": 1234567,
                        "CallSign": "TEST123",
                    },
                },
            ]
            for event in events:
                payload = {"type": "event", "source": "local-test-fixture", "time": now, **event}
                writer.write(("data: " + json.dumps(payload) + "\n\n").encode())
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            peer_closed.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setenv("AIS_AUTOSTART_ENABLED", "1")
    monkeypatch.setenv("OPENWATERS_ENABLED", "1")
    monkeypatch.setenv("OPENWATERS_URL", f"http://127.0.0.1:{port}/stream")
    monkeypatch.setenv("AISSTREAM_API_KEY", "")
    monkeypatch.setenv("AISCAST_TOKEN", "")
    try:
        async with api.lifespan(api.app):
            for _ in range(70):
                with session_scope(api._state["sessions"]) as session:
                    fixes = PositionRepository(session).latest_in_bbox(
                        api._state["settings"].gulf_bbox
                    )
                if fixes:
                    break
                await asyncio.sleep(0.05)
            assert len(fixes) == 1
            assert fixes[0].name == "LOCAL TEST"
            assert fixes[0].ship_type == 80
            assert fixes[0].imo == 1234567
            assert fixes[0].callsign == "TEST123"
            assert not api._state["ais_worker"].done()
        await asyncio.wait_for(peer_closed.wait(), timeout=2)
        assert "ais_worker" not in api._state
    finally:
        server.close()
        await server.wait_closed()
