"""Short-gap motion is an explicit estimate, never a replacement observation."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pytest

from core.ais.motion import estimate_motion
from core.db.repository import VesselFix

NOW = dt.datetime(2026, 10, 29, 12, tzinfo=dt.UTC)


def fix(**overrides):
    values = {
        "mmsi": 422123456,
        "lon": 0.0,
        "lat": 0.0,
        "ts": NOW - dt.timedelta(seconds=120),
        "sog_kn": 10.0,
        "cog_deg": 90.0,
        "heading_deg": 0.0,
        "name": None,
        "coarse_class": None,
        "source": "openwaters",
        "plausibility_score": 1.0,
        "timestamp_basis": "provider",
        "received_at": NOW,
    }
    values.update(overrides)
    return VesselFix(**values)


def test_short_gap_uses_ground_course_and_keeps_observation():
    observed = fix()
    estimate = estimate_motion(observed, NOW)
    assert estimate.status == "estimated"
    assert estimate.reason is None
    assert estimate.elapsed_seconds == 120
    # 10 knots over 120 seconds = 617.333 m due east at the equator.
    assert estimate.lon == pytest.approx(0.005545, abs=0.000002)
    assert estimate.lat == pytest.approx(0, abs=1e-9)
    assert (observed.lon, observed.lat, observed.ts) == (0, 0, NOW - dt.timedelta(seconds=120))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"source": "synthetic"}, "synthetic"),
        ({"timestamp_basis": None}, "unknown_time"),
        ({"timestamp_basis": "receipt_fallback"}, "unknown_time"),
        ({"ts": NOW - dt.timedelta(seconds=301)}, "stale"),
        ({"ts": NOW + dt.timedelta(seconds=1)}, "future_time"),
        ({"plausibility_score": 0.4}, "suspect"),
        ({"sog_kn": None}, "missing_motion"),
        ({"cog_deg": None, "heading_deg": 90.0}, "missing_motion"),
        ({"sog_kn": float("nan")}, "invalid_motion"),
        ({"cog_deg": 360.0}, "invalid_motion"),
    ],
)
def test_unsuitable_observations_do_not_get_an_estimated_point(overrides, reason):
    estimate = estimate_motion(fix(**overrides), NOW)
    assert estimate.status == "unavailable"
    assert estimate.reason == reason
    assert estimate.lon is None and estimate.lat is None


def test_limit_is_five_minutes_and_provider_reencoding_is_not_project_simulation():
    observed = fix(ts=NOW - dt.timedelta(seconds=300), provider_reencoded=True)
    assert estimate_motion(observed, NOW).status == "estimated"
    assert (
        estimate_motion(replace(observed, ts=observed.ts - dt.timedelta(seconds=1)), NOW).reason
        == "stale"
    )


def test_stationary_report_is_a_zero_displacement_estimate():
    estimate = estimate_motion(fix(sog_kn=0), NOW)
    assert (estimate.lon, estimate.lat) == pytest.approx((0, 0), abs=1e-9)


def test_longitude_wrap_and_unknown_time_are_safe():
    estimate = estimate_motion(fix(lon=179.999), NOW)
    assert -180 <= estimate.lon < -179.9
    with pytest.raises(ValueError, match="timezone"):
        estimate_motion(fix(), NOW.replace(tzinfo=None))
