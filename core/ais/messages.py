"""Normalised AIS domain types, independent of any feed vendor.

AIS is a dirty feed. Values arrive out of range, as protocol sentinels meaning
"not available", or as outright junk (MMSI 0, latitude 91.0, NUL bytes inside
names). Every field is validated here, at the boundary, so nothing downstream
has to wonder.

The ITU-R M.1371 sentinels are the important subtlety: they are *valid*
transmissions meaning "I do not know", not corruption, and must become ``None``
rather than being stored as real measurements. A stored course of 360.0 would
silently poison heading comparisons during SAR fusion.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# ITU-R M.1371 "not available" sentinels.
LAT_UNAVAILABLE = 91.0
LON_UNAVAILABLE = 181.0
SOG_UNAVAILABLE = 102.3
COG_UNAVAILABLE = 360.0
HEADING_UNAVAILABLE = 511

# A valid MMSI is exactly 9 digits.
MMSI_MIN = 100_000_000
MMSI_MAX = 999_999_999

# Speeds above this are not physically plausible for a surface vessel and
# indicate a decode error or a spoofed track.
MAX_PLAUSIBLE_SOG_KN = 80.0

#: Coarse classes. Deliberately only four: these are separable by size and
#: shape at 10 m SAR resolution. Finer types are not, and come from AIS.
COARSE_CLASSES = ("cargo_tanker", "fishing", "service_tug", "passenger_other")

# Persian Gulf littoral MMSI MID prefixes, used to cross-check claimed flag.
GULF_MIDS = {
    403: "SAU",
    408: "BHR",
    422: "IRN",
    425: "IRQ",
    447: "KWT",
    461: "OMN",
    466: "QAT",
    470: "ARE",
    471: "ARE",
}


def coarse_class(ship_type: int | None) -> str | None:
    """Map an AIS ship-type code to one of :data:`COARSE_CLASSES`.

    Tanker and cargo are deliberately merged: at 10 m they are geometrically
    near-identical, so a model claiming to separate them is fitting noise.
    """
    if ship_type is None or not 0 <= ship_type <= 99:
        return None
    if 70 <= ship_type <= 89:  # cargo (70-79) and tanker (80-89)
        return "cargo_tanker"
    if ship_type == 30:
        return "fishing"
    if ship_type in (31, 32) or 50 <= ship_type <= 59:  # towing, tugs, service
        return "service_tug"
    return "passenger_other"


def mid_of(mmsi: int) -> int:
    """Maritime Identification Digits: the first three digits of an MMSI."""
    return mmsi // 1_000_000


def is_valid_mmsi(mmsi: object) -> bool:
    if not isinstance(mmsi, int) or isinstance(mmsi, bool):
        return False
    if not MMSI_MIN <= mmsi <= MMSI_MAX:
        return False
    # MID 201-775 covers all assigned country ranges. Anything outside is a
    # craft-associated, SAR, or otherwise non-vessel identifier.
    return 201 <= mid_of(mmsi) <= 775


def clean_text(value: object) -> str | None:
    """Strip AIS text padding and reject unusable strings.

    AIS pads with '@' and the feed sometimes carries NUL bytes, either of which
    will corrupt a database write or a UI label.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.replace("\x00", "").replace("@", " ").strip()
    return cleaned or None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    # NaN is never equal to itself; infinities are equally unusable.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def normalise_latitude(value: object) -> float | None:
    number = _finite(value)
    if number is None or number == LAT_UNAVAILABLE or not -90.0 <= number <= 90.0:
        return None
    return number


def normalise_longitude(value: object) -> float | None:
    number = _finite(value)
    if number is None or number == LON_UNAVAILABLE or not -180.0 <= number <= 180.0:
        return None
    return number


def normalise_sog(value: object) -> float | None:
    number = _finite(value)
    if number is None or number == SOG_UNAVAILABLE:
        return None
    if not 0.0 <= number <= MAX_PLAUSIBLE_SOG_KN:
        return None
    return number


def normalise_cog(value: object) -> float | None:
    number = _finite(value)
    if number is None or number == COG_UNAVAILABLE or not 0.0 <= number < 360.0:
        return None
    return number


def normalise_heading(value: object) -> float | None:
    number = _finite(value)
    if number is None or int(number) == HEADING_UNAVAILABLE:
        return None
    if not 0.0 <= number < 360.0:
        return None
    return number


@dataclass(frozen=True)
class AisPositionReport:
    """A validated AIS position. Every field here is trustworthy or ``None``."""

    mmsi: int
    ts: dt.datetime
    lon: float
    lat: float
    source: str
    sog_kn: float | None = None
    cog_deg: float | None = None
    heading_deg: float | None = None
    nav_status: int | None = None
    msg_type: int | None = None
    timestamp_basis: str | None = None
    received_at: dt.datetime | None = None
    provider_source: str | None = None
    provider_reencoded: bool | None = None

    def __post_init__(self) -> None:
        if not is_valid_mmsi(self.mmsi):
            raise ValueError(f"invalid MMSI: {self.mmsi!r}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"longitude out of range: {self.lon}")
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"latitude out of range: {self.lat}")
        if self.ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.source:
            raise ValueError("source is required for licence attribution")

    @property
    def mid(self) -> int:
        return mid_of(self.mmsi)

    @property
    def flag_hint(self) -> str | None:
        """Gulf flag implied by the MMSI, if it is a Gulf-littoral MID.

        A mismatch between this and a self-reported flag is a classic
        spoofing signal.
        """
        return GULF_MIDS.get(self.mid)


@dataclass(frozen=True)
class VesselStatic:
    """Validated static and voyage data (AIS message types 5, 19, 24)."""

    mmsi: int
    name: str | None = None
    callsign: str | None = None
    imo: int | None = None
    ship_type: int | None = None
    length_m: float | None = None
    beam_m: float | None = None
    destination: str | None = None
    draught_m: float | None = None

    def __post_init__(self) -> None:
        if not is_valid_mmsi(self.mmsi):
            raise ValueError(f"invalid MMSI: {self.mmsi!r}")

    @property
    def coarse_class(self) -> str | None:
        return coarse_class(self.ship_type)


def dimensions_to_size(
    to_bow: object, to_stern: object, to_port: object, to_starboard: object
) -> tuple[float | None, float | None]:
    """Convert AIS reference-point dimensions A/B/C/D to (length, beam).

    Length is A+B and beam is C+D. All-zero dimensions mean "not reported",
    not a zero-length ship.
    """
    parts = [_finite(v) for v in (to_bow, to_stern, to_port, to_starboard)]
    if any(p is None or p < 0 for p in parts):
        return None, None
    bow, stern, port, starboard = parts  # type: ignore[misc]
    length = bow + stern
    beam = port + starboard
    return (length or None), (beam or None)
