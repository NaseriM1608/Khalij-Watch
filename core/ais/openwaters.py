"""Read native OpenWaters events without mistaking cached static positions for fixes."""

from __future__ import annotations

import datetime as dt
from typing import Any

from core.ais.aisstream import MAX_FUTURE_SKEW, POSITION_TYPES, STATIC_TYPES, parse_static
from core.ais.messages import (
    AisPositionReport,
    VesselStatic,
    clean_text,
    is_valid_mmsi,
    normalise_cog,
    normalise_heading,
    normalise_latitude,
    normalise_longitude,
    normalise_sog,
)


def parse_event(
    event: dict[str, Any], *, now: dt.datetime | None = None
) -> AisPositionReport | VesselStatic | None:
    """Use only the native position body; top-level coordinates may be cached.

    OpenWaters ``time`` is source event time when recent, otherwise provider
    receipt time. It is a provider timestamp, not independently verified GNSS.
    ``synthesized`` means the provider reencoded source observations, not that
    this application generated a simulated vessel.
    """
    if event.get("type") != "event":
        return None
    kind, body = event.get("msg_type"), event.get("message")
    if not isinstance(body, dict):
        return None
    mmsi = body.get("UserID", event.get("mmsi"))
    if not is_valid_mmsi(mmsi):
        return None
    if kind in STATIC_TYPES:
        return parse_static(
            {"MessageType": kind, "MetaData": {"MMSI": mmsi}, "Message": {kind: body}}
        )
    if kind not in POSITION_TYPES or body.get("Valid") is False:
        return None
    lat, lon = normalise_latitude(body.get("Latitude")), normalise_longitude(body.get("Longitude"))
    if lat is None or lon is None:
        return None
    received = now or event.get("_received_at") or dt.datetime.now(dt.UTC)
    reported = _event_time(event.get("time"))
    usable = reported is not None and reported <= received + MAX_FUTURE_SKEW
    upstream = clean_text(event.get("source"))
    return AisPositionReport(
        mmsi=mmsi,
        ts=reported if usable else received,
        lon=lon,
        lat=lat,
        source="openwaters",
        sog_kn=normalise_sog(body.get("Sog")),
        cog_deg=normalise_cog(body.get("Cog")),
        heading_deg=normalise_heading(body.get("TrueHeading")),
        nav_status=_small_int(body.get("NavigationalStatus"), 0, 15),
        msg_type=_small_int(body.get("MessageID"), 1, 27),
        timestamp_basis="provider" if usable else "receipt_fallback",
        received_at=received,
        provider_source=upstream[:64] if upstream else None,
        provider_reencoded=event.get("synthesized")
        if isinstance(event.get("synthesized"), bool)
        else None,
    )


def _event_time(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(dt.UTC) if parsed.tzinfo is not None else None
    except ValueError:
        return None


def _small_int(value: object, low: int, high: int) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
        else None
    )
