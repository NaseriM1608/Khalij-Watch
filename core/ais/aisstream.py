"""Parse the aisstream.io / aiscast wire format into domain types.

aisstream delivers pre-decoded JSON, so no NMEA decoding is needed for it.
``pyais`` is still required for raw-NMEA sources (aiscast UDP, bulk archives).

Everything vendor-specific lives here. ``core.ais.messages`` stays clean, so
swapping feeds means rewriting this module and nothing else.

Envelope shape::

    {"MessageType": "PositionReport",
     "MetaData": {"MMSI": 123456789, "ShipName": "X", "time_utc": "..."},
     "Message": {"PositionReport": {"Latitude": 26.5, ...}}}
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from core.ais.messages import (
    AisPositionReport,
    VesselStatic,
    clean_text,
    dimensions_to_size,
    is_valid_mmsi,
    normalise_cog,
    normalise_heading,
    normalise_latitude,
    normalise_longitude,
    normalise_sog,
)

#: Message types worth subscribing to. Anything else is noise for this project.
POSITION_TYPES = frozenset(
    {"PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport"}
)
STATIC_TYPES = frozenset({"ShipStaticData", "StaticDataReport"})
SUBSCRIBED_TYPES = sorted(POSITION_TYPES | STATIC_TYPES)

# Maximum clock skew tolerated on a reported timestamp. The feed occasionally
# emits timestamps far in the future, which would corrupt time-ordered queries.
MAX_FUTURE_SKEW = dt.timedelta(minutes=10)


class AisParseError(ValueError):
    """Raised when a message cannot be turned into a trustworthy record."""


def parse_go_timestamp(raw: object) -> dt.datetime | None:
    """Parse Go's time format, as emitted in ``MetaData.time_utc``.

    Example: ``2026-09-10 12:00:00.123456789 +0000 UTC``. Note the nanosecond
    precision, which ``datetime`` cannot represent and which ``strptime``
    refuses outright, so the fraction is truncated to microseconds.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    parts = raw.strip().split()
    if len(parts) < 2:
        return None
    stamp = f"{parts[0]} {parts[1]}"
    if "." in stamp:
        head, fraction = stamp.split(".", 1)
        stamp = f"{head}.{fraction[:6]}"
        fmt = "%Y-%m-%d %H:%M:%S.%f"
    else:
        fmt = "%Y-%m-%d %H:%M:%S"
    try:
        parsed = dt.datetime.strptime(stamp, fmt)
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.UTC)


def _resolve_timestamp(metadata: dict[str, Any], now: dt.datetime) -> dt.datetime:
    """Use the reported time when it is sane, otherwise fall back to now.

    A timestamp far in the future is unusable, so it is replaced rather than
    stored. Old timestamps are kept: a delayed message is still a real fix.
    """
    reported = parse_go_timestamp(metadata.get("time_utc"))
    if reported is None or reported > now + MAX_FUTURE_SKEW:
        return now
    return reported


def _payload(envelope: dict[str, Any], message_type: str) -> dict[str, Any]:
    message = envelope.get("Message")
    if not isinstance(message, dict):
        raise AisParseError("envelope has no Message object")
    body = message.get(message_type)
    if not isinstance(body, dict):
        raise AisParseError(f"Message lacks a {message_type} body")
    return body


def _extract_mmsi(body: dict[str, Any], metadata: dict[str, Any]) -> int:
    for candidate in (body.get("UserID"), metadata.get("MMSI")):
        if is_valid_mmsi(candidate):
            return int(candidate)  # type: ignore[arg-type]
    raise AisParseError(f"no valid MMSI (UserID={body.get('UserID')!r})")


def parse_position(
    envelope: dict[str, Any],
    *,
    source: str,
    now: dt.datetime | None = None,
) -> AisPositionReport:
    """Parse a position-report envelope.

    Raises :class:`AisParseError` if the message lacks a usable identity or
    position. Callers should count these rather than crash: a malformed
    message is routine on this feed.
    """
    message_type = envelope.get("MessageType")
    if message_type not in POSITION_TYPES:
        raise AisParseError(f"not a position report: {message_type!r}")

    body = _payload(envelope, message_type)
    metadata = envelope.get("MetaData") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    mmsi = _extract_mmsi(body, metadata)

    # Prefer the decoded message body; MetaData is a convenience copy.
    lat = normalise_latitude(body.get("Latitude"))
    if lat is None:
        lat = normalise_latitude(metadata.get("latitude"))
    lon = normalise_longitude(body.get("Longitude"))
    if lon is None:
        lon = normalise_longitude(metadata.get("longitude"))
    if lat is None or lon is None:
        raise AisParseError(f"no usable position for MMSI {mmsi}")

    received = now or dt.datetime.now(dt.UTC)
    reported = parse_go_timestamp(metadata.get("time_utc"))
    timestamp_basis = (
        "provider"
        if reported is not None and reported <= received + MAX_FUTURE_SKEW
        else "receipt_fallback"
    )
    return AisPositionReport(
        mmsi=mmsi,
        ts=_resolve_timestamp(metadata, received),
        lon=lon,
        lat=lat,
        source=source,
        sog_kn=normalise_sog(body.get("Sog")),
        cog_deg=normalise_cog(body.get("Cog")),
        heading_deg=normalise_heading(body.get("TrueHeading")),
        nav_status=_small_int(body.get("NavigationalStatus"), 0, 15),
        msg_type=_small_int(body.get("MessageID"), 1, 27),
        timestamp_basis=timestamp_basis,
        received_at=received,
    )


def _small_int(value: object, low: int, high: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if low <= value <= high else None


def _imo(value: object) -> int | None:
    # IMO numbers are 7 digits. The feed frequently carries 0 for "unknown".
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 1_000_000 <= value <= 9_999_999 else None


def parse_static(envelope: dict[str, Any], *, source: str = "aisstream") -> VesselStatic:
    """Parse a static/voyage envelope (types 5, 19 and 24).

    Type 24 arrives in two parts: Part A carries the name, Part B the type,
    call sign and dimensions. Each part is returned as its own sparse record
    and merged by the repository on upsert.
    """
    del source  # accepted for symmetry with parse_position; not stored here
    message_type = envelope.get("MessageType")
    if message_type not in STATIC_TYPES:
        raise AisParseError(f"not a static report: {message_type!r}")

    body = _payload(envelope, message_type)
    metadata = envelope.get("MetaData") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    mmsi = _extract_mmsi(body, metadata)
    dimension = body.get("Dimension")
    if isinstance(dimension, dict):
        length, beam = dimensions_to_size(
            dimension.get("A"),
            dimension.get("B"),
            dimension.get("C"),
            dimension.get("D"),
        )
    else:
        length, beam = None, None

    return VesselStatic(
        mmsi=mmsi,
        name=clean_text(body.get("Name")) or clean_text(metadata.get("ShipName")),
        callsign=clean_text(body.get("CallSign")),
        imo=_imo(body.get("ImoNumber")),
        ship_type=_small_int(body.get("Type"), 0, 99),
        length_m=length,
        beam_m=beam,
        destination=clean_text(body.get("Destination")),
        draught_m=_positive_float(body.get("MaximumStaticDraught")),
    )


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0.0 < number < 30.0 else None


def build_subscription(
    api_key: str,
    bboxes: list[list[list[float]]],
    *,
    message_types: list[str] | None = None,
    mmsis: list[str] | None = None,
) -> dict[str, Any]:
    """Build the subscription frame.

    It must be sent within 3 seconds of connecting or the server drops the
    socket, and ``BoundingBoxes`` is mandatory. Coordinates are
    ``[[lat, lon], [lat, lon]]`` - latitude first, unlike almost everything
    else in this codebase.
    """
    if not api_key:
        raise ValueError("aisstream requires an API key")
    if not bboxes:
        raise ValueError("aisstream requires at least one bounding box")

    frame: dict[str, Any] = {
        "APIKey": api_key,
        "BoundingBoxes": bboxes,
        "FilterMessageTypes": message_types or SUBSCRIBED_TYPES,
    }
    if mmsis:
        # The server caps a subscription at 200 MMSIs.
        if len(mmsis) > 200:
            raise ValueError(f"aisstream allows at most 200 MMSIs, got {len(mmsis)}")
        frame["FiltersShipMMSI"] = mmsis
    return frame
