"""Application settings, loaded from the environment or .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from core.db.spatial import GULF_BBOX, HORMUZ_BBOX, BBox

AISSTREAM_DEFAULT_URL = "wss://stream.aisstream.io/v0/stream"
AISCAST_DEFAULT_URL = "wss://ais.openwaters.io/v0/stream"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Avoids a dependency for something this small.

    Existing environment variables always win, so a shell override is possible.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    """Resolved configuration."""

    aisstream_key: str | None
    aisstream_url: str
    aiscast_url: str
    aiscast_token: str | None
    gulf_bbox: BBox
    hormuz_bbox: BBox
    database_url: str | None
    data_dir: Path
    openwaters_enabled: bool = False
    openwaters_url: str = "https://ais.openwaters.io/v1/stream"
    ais_autostart_enabled: bool = True

    @property
    def has_live_ais(self) -> bool:
        return bool(self.aisstream_key or self.aiscast_token)

    def describe_feeds(self) -> list[str]:
        available = []
        if self.aisstream_key:
            available.append("aisstream")
        if self.aiscast_token:
            available.append("aiscast")
        if self.openwaters_enabled:
            available.append("openwaters")
        return available


def load_settings(env_file: Path | None = None) -> Settings:
    _load_dotenv(env_file or Path(".env"))

    def bbox(name: str, fallback: BBox) -> BBox:
        raw = os.environ.get(name)
        if not raw:
            return fallback
        try:
            return BBox.parse(raw)
        except ValueError as exc:
            raise ValueError(f"{name} is malformed: {exc}") from exc

    return Settings(
        aisstream_key=os.environ.get("AISSTREAM_API_KEY") or None,
        aisstream_url=os.environ.get("AISSTREAM_WS_URL") or AISSTREAM_DEFAULT_URL,
        aiscast_url=os.environ.get("AISCAST_WS_URL") or AISCAST_DEFAULT_URL,
        aiscast_token=os.environ.get("AISCAST_TOKEN") or None,
        gulf_bbox=bbox("AOI_GULF_BBOX", GULF_BBOX),
        hormuz_bbox=bbox("AOI_HORMUZ_BBOX", HORMUZ_BBOX),
        database_url=os.environ.get("DATABASE_URL") or None,
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        openwaters_enabled=os.environ.get("OPENWATERS_ENABLED", "1").lower()
        in {"1", "true", "yes"},
        openwaters_url=os.environ.get("OPENWATERS_URL") or "https://ais.openwaters.io/v1/stream",
        ais_autostart_enabled=os.environ.get("AIS_AUTOSTART_ENABLED", "1").lower()
        in {"1", "true", "yes"},
    )
