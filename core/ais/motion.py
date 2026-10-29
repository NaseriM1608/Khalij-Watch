"""Bounded constant-ground-velocity estimates, separate from observed fixes."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Protocol

from pyproj import Geod

from core.ais.messages import MAX_PLAUSIBLE_SOG_KN

MAX_AGE_SECONDS = 300
_GEOD = Geod(ellps="WGS84")


class MotionFix(Protocol):
    lon: float
    lat: float
    ts: dt.datetime
    source: str
    timestamp_basis: str | None
    sog_kn: float | None
    cog_deg: float | None

    @property
    def is_suspect(self) -> bool: ...


@dataclass(frozen=True)
class MotionEstimate:
    status: str
    reason: str | None
    target_time: dt.datetime
    elapsed_seconds: float
    lon: float | None = None
    lat: float | None = None
    method: str = "constant_sog_cog"
    max_age_seconds: int = MAX_AGE_SECONDS


def estimate_motion(fix: MotionFix, target_time: dt.datetime) -> MotionEstimate:
    """Project SOG/COG for at most five minutes; no calibrated error claim.

    ``provider`` means the provider event's clock, not a verified GNSS fix time.
    Missing provenance and receipt-time fallback are deliberately ineligible.
    """
    if target_time.tzinfo is None or fix.ts.tzinfo is None:
        raise ValueError("timestamps must carry a timezone")
    elapsed = (target_time - fix.ts).total_seconds()
    reason = None
    if fix.source == "synthetic":
        reason = "synthetic"
    elif fix.timestamp_basis != "provider":
        reason = "unknown_time"
    elif elapsed < 0:
        reason = "future_time"
    elif elapsed > MAX_AGE_SECONDS:
        reason = "stale"
    elif fix.is_suspect:
        reason = "suspect"
    elif fix.sog_kn is None or fix.cog_deg is None:
        reason = "missing_motion"
    elif (
        not math.isfinite(fix.sog_kn)
        or not 0 <= fix.sog_kn <= MAX_PLAUSIBLE_SOG_KN
        or not math.isfinite(fix.cog_deg)
        or not 0 <= fix.cog_deg < 360
    ):
        reason = "invalid_motion"
    if reason:
        return MotionEstimate("unavailable", reason, target_time, elapsed)

    lon, lat, _ = _GEOD.fwd(fix.lon, fix.lat, fix.cog_deg, fix.sog_kn * (1852 / 3600) * elapsed)
    return MotionEstimate(
        "estimated",
        None,
        target_time,
        elapsed,
        lon,
        lat,
    )
