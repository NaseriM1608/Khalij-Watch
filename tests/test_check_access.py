"""Tests for the Phase 0 access checks.

The pure logic is tested offline; anything touching the network is marked
`network` so CI and offline work can deselect it with `-m 'not network'`.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_access import Result, mean_revisit_days


def _product(stamp: str) -> dict:
    return {"ContentDate": {"Start": stamp}}


def test_mean_revisit_days_collapses_same_pass_products():
    # Three products from one pass (seconds apart) plus one two days later.
    # A pass is a single observation, so the mean gap must be 2 days, not ~0.
    products = [
        _product("2026-09-01T02:06:00.000Z"),
        _product("2026-09-01T02:06:12.000Z"),
        _product("2026-09-01T02:06:24.000Z"),
        _product("2026-09-03T02:06:00.000Z"),
    ]

    assert mean_revisit_days(products) == pytest.approx(2.0, abs=0.01)


def test_mean_revisit_days_averages_uneven_gaps():
    # Gaps of 2 and 4 days -> mean 3.
    products = [
        _product("2026-09-01T02:00:00.000Z"),
        _product("2026-09-03T02:00:00.000Z"),
        _product("2026-09-07T02:00:00.000Z"),
    ]

    assert mean_revisit_days(products) == pytest.approx(3.0, abs=0.01)


def test_mean_revisit_days_returns_zero_without_enough_data():
    assert mean_revisit_days([]) == 0.0
    assert mean_revisit_days([_product("2026-09-01T02:00:00.000Z")]) == 0.0


def test_result_render_marks_failure_and_indents_notes():
    rendered = Result("CDSE", False, "no credentials", ["do the thing"]).render()

    assert rendered.startswith("[FAIL] CDSE: no credentials")
    assert "\n       do the thing" in rendered


def test_result_render_marks_success():
    assert Result("CDSE", True, "ok").render() == "[PASS] CDSE: ok"


@pytest.mark.network
def test_cdse_catalogue_reachable_and_gulf_is_covered():
    """The project's backbone. If this breaks, the plan changes."""
    from check_access import GULF_POINTS, query_sentinel1

    lon, lat = GULF_POINTS["Hormuz"]
    products = query_sentinel1(lon, lat, days=30)

    assert products, "no Sentinel-1 IW_GRDH products over Hormuz in 30 days"
    # Post-2026-06-29 the constellation is S1C + S1D only; S1A is retired.
    assert {p["Name"][:3] for p in products} <= {"S1C", "S1D"}
    # Every Gulf acquisition should be dual-pol VV+VH.
    assert all("DV" in p["Name"][14:18] for p in products)
    # Revisit at Hormuz measured ~2 days; allow generous drift before alarming.
    assert 0 < mean_revisit_days(products) < 5


def test_acquisition_timestamps_parse_as_utc():
    stamp = "2026-09-08T02:22:39.000Z"
    parsed = dt.datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")

    assert parsed.year == 2026 and parsed.hour == 2


# --------------------------------------------------------------------------
# Pixel-access verdicts
# --------------------------------------------------------------------------


def test_a_successful_probe_passes_with_its_evidence():
    from check_access import result_from_verdict

    result = result_from_verdict(
        {
            "ok": True,
            "reason": "ok",
            "product": "S1D_IW_GRDH_1SDV_TEST_COG.SAFE",
            "shape": [256, 256],
            "max_dn": 124,
            "seconds": 2.1,
        }
    )

    assert result.ok
    assert "256x256" in result.detail
    assert any("S1D_IW_GRDH_1SDV_TEST_COG.SAFE" in note for note in result.notes)


def test_a_wrong_secret_is_named_as_the_problem():
    from check_access import result_from_verdict

    result = result_from_verdict(
        {"ok": False, "reason": "wrong_secret", "detail": "SignatureDoesNotMatch"}
    )

    assert not result.ok
    assert "wrong_secret" in result.detail
    assert any("secret" in note for note in result.notes)


def test_an_unknown_key_points_at_expiry_and_the_key_manager():
    from check_access import result_from_verdict

    result = result_from_verdict({"ok": False, "reason": "unknown_access_key"})

    remedy = " ".join(result.notes)
    assert "expiry" in remedy
    assert "eodata-s3keysmanager.dataspace.copernicus.eu" in remedy


def test_a_probe_that_never_answers_fails_rather_than_passes():
    from check_access import result_from_verdict

    result = result_from_verdict(None, timed_out=True)

    assert not result.ok
    assert "no answer" in result.detail


def test_a_probe_that_dies_silently_surfaces_its_error_output():
    from check_access import result_from_verdict

    result = result_from_verdict(
        None,
        stderr_tail="Traceback (most recent call last):\nModuleNotFoundError: No module named rasterio\n",
    )

    assert not result.ok
    assert any("ModuleNotFoundError" in note for note in result.notes)
