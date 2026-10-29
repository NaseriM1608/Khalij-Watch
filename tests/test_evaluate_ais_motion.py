"""Causal replay measures provider positions without inventing long-horizon pairs."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from pyproj import Geod

from scripts.evaluate_ais_motion import evaluate_records, main

START = dt.datetime(2026, 10, 29, 12, tzinfo=dt.UTC)
GEOD = Geod(ellps="WGS84")


def record(
    seconds,
    *,
    received=None,
    mmsi=422123456,
    speed=10.0,
    course=90.0,
    displacement=None,
    event_id=None,
):
    lon, lat, _ = GEOD.fwd(
        56.5, 26.5, 90, (seconds if displacement is None else displacement) * 10 * 1852 / 3600
    )
    return {
        "received_at": (
            START + dt.timedelta(seconds=seconds if received is None else received)
        ).isoformat(),
        "event": {
            "type": "event",
            "id": event_id or f"{mmsi}-{seconds}",
            "time": (START + dt.timedelta(seconds=seconds)).isoformat(),
            "source": "aishub",
            "mmsi": mmsi,
            "msg_type": "PositionReport",
            "synthesized": True,
            "message": {
                "UserID": mmsi,
                "MessageID": 1,
                "Latitude": lat,
                "Longitude": lon,
                "Sog": speed,
                "Cog": course,
                "Valid": True,
            },
        },
    }


def test_constant_motion_beats_last_position_using_only_earlier_motion():
    result = evaluate_records([record(0), record(60, course=180)])
    horizon = result["horizons"]["60"]
    assert horizon["matched_pairs"] == 1
    assert horizon["motion_error_m"]["median"] < 0.1
    assert horizon["hold_last_error_m"]["median"] == pytest.approx(308.6667, abs=0.1)
    assert horizon["strata"]["turning"]["matched_pairs"] == 1
    assert result["samples"][0]["origin_cog_deg"] == 90
    assert (
        result["samples"][0]["reference_received_at"] != result["samples"][0]["origin_received_at"]
    )


def test_actual_receipt_lag_counts_toward_cap_and_five_minutes_stays_zero():
    result = evaluate_records([record(0, received=90), record(150, received=240)])
    assert result["horizons"]["60"]["matched_pairs"] == 1
    assert result["samples"][0]["observation_elapsed_seconds"] == 150
    assert result["horizons"]["300"]["matched_pairs"] == 0
    assert result["horizons"]["300"]["skipped"]["stale"] == 2
    assert result["horizons"]["300"]["motion_error_m"]["median"] is None


def test_reference_is_at_or_after_target_with_at_most_fifteen_seconds_slack():
    assert evaluate_records([record(0), record(59)])["horizons"]["60"]["matched_pairs"] == 0
    result = evaluate_records([record(0), record(70)])
    assert result["horizons"]["60"]["matched_pairs"] == 1
    assert result["samples"][0]["observation_elapsed_seconds"] == 70
    assert evaluate_records([record(0), record(76)])["horizons"]["60"]["matched_pairs"] == 0


def test_dedup_uses_id_and_time_not_id_alone_and_stationary_is_separate():
    first = record(0, speed=0, displacement=0, event_id="reused")
    second = record(60, speed=0, displacement=0, event_id="reused")
    result = evaluate_records([first, first, second])
    assert result["input_counts"]["duplicates"] == 1
    assert result["horizons"]["60"]["matched_pairs"] == 1
    assert result["horizons"]["60"]["strata"]["stationary"]["matched_pairs"] == 1


def test_future_time_and_late_old_inputs_cannot_replace_available_latest_fix():
    result = evaluate_records(
        [
            record(120, received=130),
            record(0, received=0),
            record(60, received=90),
            record(30, received=100),
            record(500, received=20),
        ]
    )
    assert result["input_counts"]["future_time"] == 1
    assert result["input_counts"]["out_of_order_origins"] == 1
    assert all(
        s["origin_time"] <= s["origin_received_at"] < s["reference_received_at"]
        for s in result["samples"]
    )


def test_unknown_time_static_and_missing_motion_are_counted_without_fabricated_errors():
    bad_time = record(0)
    bad_time["event"]["time"] = "bad"
    static = record(0, event_id="static")
    static["event"]["msg_type"] = "ShipStaticData"
    static["event"]["message"] = {"UserID": 422123456, "Name": "NAME"}
    result = evaluate_records([bad_time, static, record(1, speed=None)])
    assert result["input_counts"]["unknown_time"] == 1
    assert result["input_counts"]["static"] == 1
    assert result["input_counts"]["missing_motion"] == 1
    assert result["horizons"]["60"]["matched_pairs"] == 0


def test_cli_writes_explicit_output_and_counts_partial_json_line(tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text(
        json.dumps(record(0)) + "\n" + json.dumps(record(60)) + '\n{"partial":', encoding="utf-8"
    )
    target = tmp_path / "result.json"
    assert main(["--input", str(source), "--out", str(target)]) == 0
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["malformed_json_lines"] == 1
    assert result["horizons"]["60"]["matched_pairs"] == 1
