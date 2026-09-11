"""Tests for AIS plausibility scoring.

Two halves matter equally:
  - the spoofing signatures actually observed around Hormuz are caught, and
  - ordinary Gulf traffic is NOT flagged. A filter that cries wolf on vessels
    berthed in Jebel Ali is worse than no filter at all.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.ais.landmask import NoLandMask, PolygonLandMask, point_in_polygon
from core.ais.messages import AisPositionReport
from core.ais.plausibility import (
    MAX_PLAUSIBLE_SPEED_KN,
    Anomaly,
    PlausibilityAnalyzer,
    check_collinear_track,
    score_from,
)
from core.geo import (
    angular_difference_deg,
    haversine_m,
    linear_deviation_m,
    speed_kn,
)

MMSI = 422123456
NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)


def fix(lon=56.5, lat=26.5, seconds=0, mmsi=MMSI, **kwargs) -> AisPositionReport:
    return AisPositionReport(
        mmsi=mmsi,
        ts=NOW + dt.timedelta(seconds=seconds),
        lon=lon,
        lat=lat,
        source="test",
        **kwargs,
    )


@pytest.fixture
def analyzer():
    return PlausibilityAnalyzer()


@pytest.fixture
def sea_analyzer():
    """Analyzer with the land check disabled, to isolate kinematics."""
    return PlausibilityAnalyzer(land_mask=NoLandMask())


# --------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------


def test_haversine_matches_a_known_distance():
    # Bandar Abbas to Dubai: about 2.0 deg of latitude plus 1.0 deg of
    # longitude at 26 deg N, so roughly 240 km.
    metres = haversine_m(56.28, 27.18, 55.27, 25.20)
    assert 230_000 < metres < 255_000


def test_zero_distance_is_zero():
    assert haversine_m(56.5, 26.5, 56.5, 26.5) == pytest.approx(0.0, abs=1e-6)


def test_angular_difference_wraps_around_north():
    """359 deg and 1 deg are 2 apart, not 358."""
    assert angular_difference_deg(359.0, 1.0) == pytest.approx(2.0)
    assert angular_difference_deg(1.0, 359.0) == pytest.approx(2.0)
    assert angular_difference_deg(0.0, 180.0) == pytest.approx(180.0)


def test_speed_from_distance_and_time():
    # One nautical mile in one hour is one knot.
    assert speed_kn(1852.0, 3600.0) == pytest.approx(1.0, abs=1e-6)


def test_speed_of_zero_duration_is_infinite_when_it_moved():
    assert speed_kn(1000.0, 0.0) == float("inf")
    assert speed_kn(0.0, 0.0) == 0.0


def test_linear_deviation_is_zero_on_the_line():
    """Constant latitude counts as straight, which a great circle would not."""
    assert linear_deviation_m(55.0, 26.0, 54.0, 26.0, 56.0, 26.0) < 1.0


def test_linear_deviation_measures_offset_from_the_line():
    offset = linear_deviation_m(55.0, 26.1, 54.0, 26.0, 56.0, 26.0)
    # 0.1 deg of latitude is about 11 km.
    assert 10_000 < offset < 12_000


# --------------------------------------------------------------------------
# Land mask
# --------------------------------------------------------------------------


def test_point_in_polygon_basic():
    square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    assert point_in_polygon(1.0, 1.0, square)
    assert not point_in_polygon(3.0, 1.0, square)


@pytest.mark.parametrize(
    ("lon", "lat", "place"),
    [
        (52.53, 29.60, "Shiraz, deep in the Iranian interior"),
        (46.72, 24.69, "Riyadh"),
        (55.35, 23.10, "Al Ain / Omani interior"),
        (57.53, 22.93, "Nizwa, Omani interior"),
    ],
)
def test_inland_cities_are_recognised_as_land(lon, lat, place):
    assert PolygonLandMask().is_land(lon, lat), place


@pytest.mark.parametrize(
    ("lon", "lat", "place"),
    [
        (56.50, 26.50, "mid-strait"),
        (52.00, 27.00, "central Gulf"),
        (55.03, 24.98, "Jebel Ali anchorage"),
        (56.40, 25.18, "Fujairah anchorage"),
        (56.28, 27.18, "Bandar Abbas approaches"),
        (50.16, 26.70, "Ras Tanura terminal"),
        (51.53, 25.29, "Doha, on the Qatari coast"),
        (48.80, 29.60, "northern Gulf off Kuwait"),
        # Regression: the Omani polygon once reached 25.6 N and marked the open
        # Gulf of Oman, the approach to Hormuz, as land.
        (57.90, 24.80, "open Gulf of Oman"),
        (57.35, 25.15, "western Gulf of Oman"),
        (58.50, 24.50, "Gulf of Oman north of Muscat"),
        (58.57, 23.63, "Muscat, Sultan Qaboos port"),
        (56.73, 24.36, "Sohar port"),
    ],
)
def test_water_and_ports_are_never_flagged_as_land(lon, lat, place):
    """A false positive here would discredit the whole filter."""
    assert not PolygonLandMask().is_land(lon, lat), place


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_score_is_one_when_nothing_is_wrong():
    assert score_from([]) == 1.0


def test_penalties_compound_multiplicatively():
    combined = score_from([Anomaly("A", 0.5, ""), Anomaly("B", 0.5, "")])
    assert combined == pytest.approx(0.25)


def test_score_is_clamped_to_zero():
    assert score_from([Anomaly("A", 1.0, ""), Anomaly("B", 0.9, "")]) == 0.0


# --------------------------------------------------------------------------
# Kinematic checks
# --------------------------------------------------------------------------


def test_ordinary_transit_is_not_flagged(sea_analyzer):
    """A vessel making 12 kn on a steady course must score clean."""
    # 12 kn is about 111 m per 18 s.
    verdict = None
    for step in range(6):
        verdict = sea_analyzer.evaluate(
            fix(
                lon=56.50 + step * 0.0015,
                lat=26.50 + step * 0.0011,
                seconds=step * 20,
                sog_kn=12.0,
                cog_deg=54.0,
            )
        )
    assert verdict.score == 1.0
    assert verdict.anomalies == ()


def test_teleport_is_flagged(sea_analyzer):
    sea_analyzer.evaluate(fix(lon=56.5, lat=26.5, seconds=0))
    verdict = sea_analyzer.evaluate(fix(lon=51.0, lat=27.5, seconds=30))

    assert "TELEPORT" in verdict.codes
    assert verdict.is_suspect


def test_impossible_speed_is_flagged(sea_analyzer):
    sea_analyzer.evaluate(fix(lon=56.50, lat=26.50, seconds=0))
    # ~10 km in 60 s is roughly 320 kn.
    verdict = sea_analyzer.evaluate(fix(lon=56.60, lat=26.50, seconds=60))

    assert "IMPOSSIBLE_SPEED" in verdict.codes


def test_speed_just_under_the_limit_is_accepted(sea_analyzer):
    """Guards the threshold against off-by-one drift."""
    sea_analyzer.evaluate(fix(lon=56.5, lat=26.5, seconds=0))
    # Travel a distance implying just under the cap over one hour.
    metres = (MAX_PLAUSIBLE_SPEED_KN - 5) * 1852.0
    degrees = metres / 111_320.0
    verdict = sea_analyzer.evaluate(fix(lon=56.5, lat=26.5 + degrees, seconds=3600))

    assert "IMPOSSIBLE_SPEED" not in verdict.codes


def test_duplicate_identity_is_flagged(sea_analyzer):
    """The same MMSI in two places within seconds."""
    sea_analyzer.evaluate(fix(lon=56.5, lat=26.5, seconds=0))
    verdict = sea_analyzer.evaluate(fix(lon=56.8, lat=26.9, seconds=10))

    assert "DUPLICATE_IDENTITY" in verdict.codes


def test_impossible_turn_is_flagged(sea_analyzer):
    sea_analyzer.evaluate(fix(seconds=0, cog_deg=0.0, sog_kn=10.0))
    verdict = sea_analyzer.evaluate(
        fix(lon=56.501, lat=26.501, seconds=5, cog_deg=170.0, sog_kn=10.0)
    )

    assert "IMPOSSIBLE_TURN" in verdict.codes


def test_slow_vessel_heading_swing_is_not_a_turn(sea_analyzer):
    """At anchor a vessel swings freely; that is not a manoeuvre."""
    sea_analyzer.evaluate(fix(seconds=0, cog_deg=0.0, sog_kn=0.1))
    verdict = sea_analyzer.evaluate(
        fix(lon=56.5001, lat=26.5001, seconds=5, cog_deg=170.0, sog_kn=0.1)
    )

    assert "IMPOSSIBLE_TURN" not in verdict.codes


def test_course_change_across_north_is_not_a_turn(sea_analyzer):
    """359 -> 1 degrees is a 2 degree change, not 358."""
    sea_analyzer.evaluate(fix(seconds=0, cog_deg=359.0, sog_kn=10.0))
    verdict = sea_analyzer.evaluate(
        fix(lon=56.5005, lat=26.5005, seconds=5, cog_deg=1.0, sog_kn=10.0)
    )

    assert "IMPOSSIBLE_TURN" not in verdict.codes


def test_reported_speed_disagreeing_with_track_is_flagged(sea_analyzer):
    """Claims to be stopped while the track shows it moving fast."""
    sea_analyzer.evaluate(fix(lon=56.50, lat=26.50, seconds=0, sog_kn=0.0))
    verdict = sea_analyzer.evaluate(fix(lon=56.55, lat=26.50, seconds=600, sog_kn=0.0))

    assert "SPEED_MISMATCH" in verdict.codes


def test_anchored_vessel_is_not_a_speed_mismatch(sea_analyzer):
    """Stationary and reporting zero: entirely consistent."""
    sea_analyzer.evaluate(fix(lon=56.40, lat=25.18, seconds=0, sog_kn=0.0))
    verdict = sea_analyzer.evaluate(fix(lon=56.4002, lat=25.1802, seconds=300, sog_kn=0.0))

    assert "SPEED_MISMATCH" not in verdict.codes


# --------------------------------------------------------------------------
# Track shape
# --------------------------------------------------------------------------


def test_perfectly_straight_track_is_flagged():
    """A synthesised track interpolates; a real one wanders."""
    history = [fix(lon=54.0 + i * 0.1, lat=26.0, seconds=i * 600) for i in range(8)]
    # Latitude is held exactly constant, so points sit on one line.
    anomaly = check_collinear_track(history)

    assert anomaly is not None
    assert anomaly.code == "COLLINEAR_TRACK"


def test_wandering_track_is_not_flagged():
    wobble = [0.0, 0.02, -0.03, 0.04, -0.02, 0.03, -0.04, 0.01]
    history = [fix(lon=54.0 + i * 0.1, lat=26.0 + wobble[i], seconds=i * 600) for i in range(8)]

    assert check_collinear_track(history) is None


def test_short_track_is_not_judged_for_straightness():
    history = [fix(lon=54.0 + i * 0.1, lat=26.0, seconds=i * 600) for i in range(3)]

    assert check_collinear_track(history) is None


def test_short_distance_track_is_not_judged_for_straightness():
    """A vessel holding station produces a tiny, technically straight track."""
    history = [fix(lon=54.0 + i * 0.0005, lat=26.0, seconds=i * 600) for i in range(8)]

    assert check_collinear_track(history) is None


# --------------------------------------------------------------------------
# Land integration
# --------------------------------------------------------------------------


def test_position_on_land_is_flagged(analyzer):
    verdict = analyzer.evaluate(fix(lon=52.53, lat=29.60))  # Shiraz

    assert "ON_LAND" in verdict.codes
    assert verdict.is_suspect


def test_position_at_sea_is_clean(analyzer):
    verdict = analyzer.evaluate(fix(lon=56.5, lat=26.5))

    assert verdict.score == 1.0


# --------------------------------------------------------------------------
# State management
# --------------------------------------------------------------------------


def test_vessels_are_tracked_independently(sea_analyzer):
    """One vessel's history must not contaminate another's."""
    sea_analyzer.evaluate(fix(lon=56.5, lat=26.5, seconds=0, mmsi=422111111))
    verdict = sea_analyzer.evaluate(fix(lon=51.0, lat=27.5, seconds=10, mmsi=470222222))

    assert verdict.anomalies == ()


def test_history_is_capped_to_bound_memory():
    analyzer = PlausibilityAnalyzer(land_mask=NoLandMask(), max_vessels=3)
    for index in range(10):
        analyzer.evaluate(fix(mmsi=422000000 + index))

    assert analyzer.tracked_vessels == 3


def test_flags_serialise_for_storage(analyzer):
    verdict = analyzer.evaluate(fix(lon=52.53, lat=29.60))
    flags = verdict.as_flags()

    assert flags is not None
    assert flags[0]["code"] == "ON_LAND"
    assert "detail" in flags[0]


def test_clean_position_stores_no_flags(analyzer):
    assert analyzer.evaluate(fix(lon=56.5, lat=26.5)).as_flags() is None


# --------------------------------------------------------------------------
# Calibration: realistic traffic must survive the filter
# --------------------------------------------------------------------------


def _realistic_transit(steps: int, interval_s: int, speed_kn_: float, rng_seed: int):
    """A vessel on a steady course with realistic GPS noise."""
    import random

    rng = random.Random(rng_seed)
    noise_deg = 0.0003  # about 33 m, typical AIS position noise
    reports = []
    for step in range(steps):
        hours = (step * interval_s) / 3600.0
        # Northbound: 1 degree of latitude is 60 nautical miles.
        lat = 26.0 + (speed_kn_ * hours) / 60.0
        reports.append(
            fix(
                lon=54.0 + rng.uniform(-noise_deg, noise_deg),
                lat=lat + rng.uniform(-noise_deg, noise_deg),
                seconds=step * interval_s,
                sog_kn=speed_kn_,
                cog_deg=0.0,
            )
        )
    return reports


def test_a_vessel_in_a_traffic_lane_is_not_flagged(sea_analyzer):
    """The critical false-positive case.

    A ship on autopilot in a separation scheme travels a near-straight line for
    hours. That is ordinary navigation, and flagging it would discredit the
    whole filter. Straightness alone must never be enough.
    """
    verdicts = [
        sea_analyzer.evaluate(report)
        for report in _realistic_transit(steps=12, interval_s=300, speed_kn_=13.0, rng_seed=7)
    ]

    flagged = [v for v in verdicts if v.anomalies]
    assert flagged == [], f"legitimate transit was flagged: {[v.codes for v in flagged]}"


def test_an_impossibly_straight_track_is_still_caught(sea_analyzer):
    """Zero deviation is below the GPS noise floor, so it cannot be real."""
    verdicts = []
    for step in range(12):
        hours = (step * 300) / 3600.0
        verdicts.append(
            sea_analyzer.evaluate(
                fix(
                    lon=54.0,  # no noise at all: mathematically impossible
                    lat=26.0 + (13.0 * hours) / 60.0,
                    seconds=step * 300,
                    sog_kn=13.0,
                    cog_deg=0.0,
                )
            )
        )

    assert any("COLLINEAR_TRACK" in v.codes for v in verdicts)


def test_speed_mismatch_ignores_gaps_that_are_too_short(sea_analyzer):
    """Over a few seconds, position noise dominates the implied speed."""
    sea_analyzer.evaluate(fix(lon=54.0, lat=26.0, seconds=0, sog_kn=13.0))
    verdict = sea_analyzer.evaluate(fix(lon=54.0001, lat=26.0001, seconds=5, sog_kn=13.0))

    assert "SPEED_MISMATCH" not in verdict.codes


def test_speed_mismatch_ignores_gaps_that_are_too_long(sea_analyzer):
    """Over an hour a vessel may legitimately have changed speed."""
    sea_analyzer.evaluate(fix(lon=54.0, lat=26.0, seconds=0, sog_kn=13.0))
    verdict = sea_analyzer.evaluate(fix(lon=54.0, lat=26.01, seconds=3600, sog_kn=13.0))

    assert "SPEED_MISMATCH" not in verdict.codes
