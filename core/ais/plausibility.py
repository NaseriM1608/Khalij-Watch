"""Score AIS positions for plausibility.

Why this is a first-class ingest stage rather than an afterthought
-----------------------------------------------------------------
In this AOI, AIS is adversarial. Through 2026, GNSS jamming and deliberate
spoofing around the Strait of Hormuz were widespread, with reporting claiming
that nearly two of every three logged August 2026 transits were false.
Rendering AIS as ground truth would render fiction.

Policy: **flag and score, never silently drop.** A fabricated track is itself
evidence - it is a vessel actively concealing its movement - so it is stored
with its anomalies attached and the operator decides what to do with it.

The kinematic checks carry most of the signal and need no external data. The
land check needs a mask and is deliberately conservative (see
:mod:`core.ais.landmask`).
"""

from __future__ import annotations

import datetime as dt
from collections import OrderedDict
from dataclasses import dataclass, field

from core.ais.landmask import DEFAULT_LAND_MASK, LandMask
from core.ais.messages import GULF_MIDS, AisPositionReport
from core.geo import (
    angular_difference_deg,
    bearing_deg,
    haversine_m,
    linear_deviation_m,
    speed_kn,
)

# --- thresholds ------------------------------------------------------------

# Fast patrol craft reach ~50 kn; commercial traffic stays under 30. Above
# this, the pair of positions cannot both be true.
MAX_PLAUSIBLE_SPEED_KN = 60.0

# A jump this large between consecutive fixes is a teleport regardless of the
# gap, and is the classic signature of an identity being reused elsewhere.
TELEPORT_DISTANCE_M = 100_000.0
TELEPORT_WINDOW_S = 900.0

# Ships turn slowly. Even a tug rarely exceeds a few degrees per second.
MAX_TURN_RATE_DEG_S = 8.0
MIN_TURN_SPEED_KN = 1.0  # below this, heading is noise, not a turn

# The vessel's own reported speed should roughly agree with the speed implied
# by its consecutive positions.
SPEED_MISMATCH_FACTOR = 4.0
SPEED_MISMATCH_FLOOR_KN = 5.0
# Reported speed is instantaneous; implied speed is an average over the gap.
# Comparing them is only fair in a middle band. Too short and GPS position
# noise dominates the displacement; too long and a vessel that legitimately
# altered speed looks like a liar.
MIN_MISMATCH_GAP_S = 60.0
MAX_MISMATCH_GAP_S = 900.0

# Straightness alone is NOT evidence: a vessel on autopilot in a traffic
# separation scheme genuinely travels a straight line for hours. What no real
# vessel can do is travel a line with essentially ZERO deviation, because GPS
# noise, yaw and set-and-drift always contribute tens of metres. So the
# tolerance sits below the real-world noise floor: this asks "is this track
# mathematically impossible?", not "is this track straight?".
COLLINEAR_MIN_POINTS = 8
COLLINEAR_TOLERANCE_M = 5.0
COLLINEAR_MIN_LENGTH_M = 20_000.0

# Two fixes for one MMSI, far apart, at nearly the same instant.
DUPLICATE_WINDOW_S = 60.0
DUPLICATE_DISTANCE_M = 5_000.0

# How many recent fixes to retain per vessel for track-shape analysis.
HISTORY_LENGTH = 12
# Cap on tracked vessels, so a busy feed cannot grow memory without bound.
MAX_TRACKED_VESSELS = 20_000


@dataclass(frozen=True)
class Anomaly:
    """One reason a position is doubted."""

    code: str
    penalty: float
    detail: str


@dataclass(frozen=True)
class Plausibility:
    """The verdict on one position."""

    score: float
    anomalies: tuple[Anomaly, ...]

    @property
    def codes(self) -> list[str]:
        return [a.code for a in self.anomalies]

    @property
    def is_suspect(self) -> bool:
        return self.score < 0.5

    def as_flags(self) -> list[dict[str, object]] | None:
        """JSON form for storage. ``None`` when nothing was wrong."""
        if not self.anomalies:
            return None
        return [
            {"code": a.code, "penalty": round(a.penalty, 3), "detail": a.detail}
            for a in self.anomalies
        ]


PLAUSIBLE = Plausibility(score=1.0, anomalies=())


@dataclass(frozen=True)
class ScoredPosition:
    """A position together with its plausibility verdict.

    Kept as a wrapper rather than fields on ``AisPositionReport`` so the parsed
    message stays a faithful record of what was broadcast, separate from our
    judgement of it.
    """

    report: AisPositionReport
    plausibility: Plausibility


@dataclass
class _VesselHistory:
    """Recent fixes for one MMSI."""

    fixes: list[AisPositionReport] = field(default_factory=list)

    def add(self, report: AisPositionReport) -> None:
        self.fixes.append(report)
        if len(self.fixes) > HISTORY_LENGTH:
            del self.fixes[0]

    @property
    def last(self) -> AisPositionReport | None:
        return self.fixes[-1] if self.fixes else None


def _seconds_between(a: dt.datetime, b: dt.datetime) -> float:
    return abs((b - a).total_seconds())


def check_on_land(report: AisPositionReport, mask: LandMask) -> Anomaly | None:
    if not mask.is_land(report.lon, report.lat):
        return None
    return Anomaly(
        code="ON_LAND",
        penalty=0.7,
        detail=f"position {report.lat:.3f},{report.lon:.3f} is well inland",
    )


def check_impossible_speed(
    previous: AisPositionReport, current: AisPositionReport
) -> Anomaly | None:
    gap = _seconds_between(previous.ts, current.ts)
    if gap <= 0:
        return None
    distance = haversine_m(previous.lon, previous.lat, current.lon, current.lat)
    implied = speed_kn(distance, gap)
    if implied <= MAX_PLAUSIBLE_SPEED_KN:
        return None
    return Anomaly(
        code="IMPOSSIBLE_SPEED",
        penalty=0.6,
        detail=f"implied {implied:.0f} kn over {distance / 1000:.1f} km in {gap:.0f}s",
    )


def check_teleport(previous: AisPositionReport, current: AisPositionReport) -> Anomaly | None:
    gap = _seconds_between(previous.ts, current.ts)
    if gap > TELEPORT_WINDOW_S:
        return None
    distance = haversine_m(previous.lon, previous.lat, current.lon, current.lat)
    if distance < TELEPORT_DISTANCE_M:
        return None
    return Anomaly(
        code="TELEPORT",
        penalty=0.8,
        detail=f"jumped {distance / 1000:.0f} km in {gap:.0f}s",
    )


def check_duplicate_identity(
    previous: AisPositionReport, current: AisPositionReport
) -> Anomaly | None:
    """Two distant fixes at nearly the same instant: the MMSI is in two places."""
    gap = _seconds_between(previous.ts, current.ts)
    if gap > DUPLICATE_WINDOW_S:
        return None
    distance = haversine_m(previous.lon, previous.lat, current.lon, current.lat)
    if distance < DUPLICATE_DISTANCE_M:
        return None
    return Anomaly(
        code="DUPLICATE_IDENTITY",
        penalty=0.8,
        detail=f"two fixes {distance / 1000:.0f} km apart within {gap:.0f}s",
    )


def check_turn_rate(previous: AisPositionReport, current: AisPositionReport) -> Anomaly | None:
    if previous.cog_deg is None or current.cog_deg is None:
        return None
    # Below steerage way a reported course is noise, not a manoeuvre.
    if (current.sog_kn or 0.0) < MIN_TURN_SPEED_KN:
        return None
    gap = _seconds_between(previous.ts, current.ts)
    if gap <= 0:
        return None
    change = angular_difference_deg(previous.cog_deg, current.cog_deg)
    rate = change / gap
    if rate <= MAX_TURN_RATE_DEG_S:
        return None
    return Anomaly(
        code="IMPOSSIBLE_TURN",
        penalty=0.4,
        detail=f"{change:.0f} deg in {gap:.0f}s ({rate:.1f} deg/s)",
    )


def check_reported_speed(previous: AisPositionReport, current: AisPositionReport) -> Anomaly | None:
    """The vessel's own speed should agree with the speed its track implies."""
    if current.sog_kn is None:
        return None
    gap = _seconds_between(previous.ts, current.ts)
    if not MIN_MISMATCH_GAP_S <= gap <= MAX_MISMATCH_GAP_S:
        return None
    distance = haversine_m(previous.lon, previous.lat, current.lon, current.lat)
    implied = speed_kn(distance, gap)
    reported = current.sog_kn
    # When both are slow there is no meaningful disagreement: a vessel swinging
    # at anchor reports 0 kn while drifting a few metres, and that is correct.
    if implied < SPEED_MISMATCH_FLOOR_KN and reported < SPEED_MISMATCH_FLOOR_KN:
        return None
    higher = max(implied, reported)
    lower = max(min(implied, reported), 0.1)
    if higher / lower <= SPEED_MISMATCH_FACTOR:
        return None
    return Anomaly(
        code="SPEED_MISMATCH",
        penalty=0.35,
        detail=f"reports {reported:.1f} kn but track implies {implied:.1f} kn",
    )


def check_collinear_track(history: list[AisPositionReport]) -> Anomaly | None:
    """Detect a track that is implausibly straight.

    Real vessels yaw, drift and adjust. A synthesised track frequently
    interpolates between two endpoints, leaving every intermediate point on a
    single great circle to within a few metres.
    """
    if len(history) < COLLINEAR_MIN_POINTS:
        return None
    start, end = history[0], history[-1]
    span = haversine_m(start.lon, start.lat, end.lon, end.lat)
    if span < COLLINEAR_MIN_LENGTH_M:
        return None

    worst = 0.0
    for point in history[1:-1]:
        offset = linear_deviation_m(point.lon, point.lat, start.lon, start.lat, end.lon, end.lat)
        worst = max(worst, offset)
    if worst > COLLINEAR_TOLERANCE_M:
        return None
    return Anomaly(
        code="COLLINEAR_TRACK",
        # Weighted as strong evidence, not a hint. The tolerance sits BELOW
        # the real-world GPS noise floor, so this does not fire on a vessel
        # merely holding a steady course - it fires on a track that no real
        # receiver could have produced.
        penalty=0.55,
        detail=(
            f"{len(history)} fixes over {span / 1000:.0f} km deviate at most "
            f"{worst:.1f} m from a straight line - below real GPS noise"
        ),
    )


def check_mid_plausibility(report: AisPositionReport) -> Anomaly | None:
    """A Gulf-flagged MMSI seen far outside any plausible Gulf position.

    This is weak evidence on its own, so it carries a small penalty. It exists
    mostly to surface identity reuse.
    """
    if report.mid not in GULF_MIDS:
        return None
    # Generous: the Gulf plus its wider approaches.
    if 40.0 <= report.lon <= 70.0 and 10.0 <= report.lat <= 35.0:
        return None
    return Anomaly(
        code="MID_REGION_MISMATCH",
        penalty=0.15,
        detail=f"MID {report.mid} ({GULF_MIDS[report.mid]}) reporting far from the Gulf",
    )


def score_from(anomalies: list[Anomaly]) -> float:
    """Combine penalties multiplicatively.

    Multiplicative rather than additive so that several weak signals compound
    into real doubt without any single one saturating the score.
    """
    score = 1.0
    for anomaly in anomalies:
        score *= 1.0 - min(max(anomaly.penalty, 0.0), 1.0)
    return round(max(0.0, min(1.0, score)), 4)


class PlausibilityAnalyzer:
    """Stateful analyser holding recent history per vessel.

    History is bounded twice over: a fixed number of fixes per vessel, and an
    LRU cap on tracked vessels, so a busy feed cannot grow memory unbounded.
    """

    def __init__(
        self,
        *,
        land_mask: LandMask | None = None,
        max_vessels: int = MAX_TRACKED_VESSELS,
    ) -> None:
        self._land_mask = land_mask if land_mask is not None else DEFAULT_LAND_MASK
        self._max_vessels = max_vessels
        self._history: OrderedDict[int, _VesselHistory] = OrderedDict()

    @property
    def tracked_vessels(self) -> int:
        return len(self._history)

    def evaluate(self, report: AisPositionReport) -> Plausibility:
        """Score a position, then fold it into that vessel's history."""
        anomalies: list[Anomaly] = []

        on_land = check_on_land(report, self._land_mask)
        if on_land:
            anomalies.append(on_land)

        mid_issue = check_mid_plausibility(report)
        if mid_issue:
            anomalies.append(mid_issue)

        history = self._history.get(report.mmsi)
        previous = history.last if history else None
        if previous is not None:
            for check in (
                check_teleport,
                check_duplicate_identity,
                check_impossible_speed,
                check_turn_rate,
                check_reported_speed,
            ):
                found = check(previous, report)
                if found:
                    anomalies.append(found)

        self._remember(report)
        current = self._history[report.mmsi]
        collinear = check_collinear_track(current.fixes)
        if collinear:
            anomalies.append(collinear)

        if not anomalies:
            return PLAUSIBLE
        return Plausibility(score=score_from(anomalies), anomalies=tuple(anomalies))

    def _remember(self, report: AisPositionReport) -> None:
        history = self._history.get(report.mmsi)
        if history is None:
            history = _VesselHistory()
            self._history[report.mmsi] = history
        history.add(report)
        self._history.move_to_end(report.mmsi)
        while len(self._history) > self._max_vessels:
            self._history.popitem(last=False)


def bearing_change(previous: AisPositionReport, current: AisPositionReport) -> float | None:
    """Course change implied by two positions, for diagnostics."""
    if previous.cog_deg is None:
        return None
    implied = bearing_deg(previous.lon, previous.lat, current.lon, current.lat)
    return angular_difference_deg(previous.cog_deg, implied)
