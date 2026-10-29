"""Causal short-gap motion validation against later provider AIS positions.

This is a descriptive replay, not independent location truth or fitted accuracy.
Only collector JSONL is read; the application database and network are untouched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ais.messages import AisPositionReport, VesselStatic  # noqa: E402
from core.ais.motion import estimate_motion  # noqa: E402
from core.ais.openwaters import parse_event  # noqa: E402
from core.ais.plausibility import PlausibilityAnalyzer  # noqa: E402
from core.db.repository import VesselFix  # noqa: E402

HORIZONS = (60, 180, 300)
REFERENCE_SLACK_SECONDS = 15
_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class Observation:
    fix: VesselFix
    received: dt.datetime


def _time(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(dt.UTC) if parsed.tzinfo is not None else None
    except ValueError:
        return None


def _distribution(values: Sequence[float]) -> dict:
    if not values:
        return {"median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.9
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower]),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _summary(samples: Sequence[dict]) -> dict:
    return {
        "matched_pairs": len(samples),
        "distinct_vessels": len({s["mmsi"] for s in samples}),
        "motion_error_m": _distribution([s["motion_error_m"] for s in samples]),
        "hold_last_error_m": _distribution([s["hold_last_error_m"] for s in samples]),
    }


def evaluate_records(records: Iterable[dict]) -> dict:
    """Make predictions at receipt time; later positions are reference-only.

    For each horizon, find the earliest same-vessel provider fix in
    [receipt+horizon, receipt+horizon+15s]. Project to that reference's actual
    provider time, count receipt lag toward the existing 300-second cap, and
    compare both projection and hold-last to the same later reported position.
    """
    counts = Counter(
        dict.fromkeys(
            (
                "records",
                "duplicates",
                "malformed",
                "static",
                "unknown_time",
                "future_time",
                "synthetic",
                "suspect",
                "missing_motion",
                "invalid_motion",
                "stale",
                "out_of_order_origins",
                "position_reports",
                "eligible_origins",
            ),
            0,
        )
    )
    ordered = []
    seen = set()
    for row in records:
        counts["records"] += 1
        if not isinstance(row, dict) or not isinstance(row.get("event"), dict):
            counts["malformed"] += 1
            continue
        received, event = _time(row.get("received_at")), row["event"]
        if received is None:
            counts["malformed"] += 1
            continue
        key = (
            json.dumps([event.get("id"), event.get("time")])
            if event.get("id") is not None
            else json.dumps(event, sort_keys=True)
        )
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        ordered.append((received, event))
    ordered.sort(key=lambda item: item[0])

    analyzer = PlausibilityAnalyzer()
    references: dict[int, list[Observation]] = defaultdict(list)
    origins: list[Observation] = []
    latest: dict[int, dt.datetime] = {}
    source_counts = Counter()
    lags, gaps = [], []
    for received, event in ordered:
        parsed = parse_event(event, now=received)
        if isinstance(parsed, VesselStatic):
            counts["static"] += 1
            continue
        if not isinstance(parsed, AisPositionReport):
            counts["malformed"] += 1
            continue
        counts["position_reports"] += 1
        source_counts[parsed.provider_source or "unknown"] += 1
        if parsed.provider_source == "synthetic":
            counts["synthetic"] += 1
            continue
        if parsed.timestamp_basis != "provider":
            counts["unknown_time"] += 1
            continue
        if parsed.ts > received:
            counts["future_time"] += 1
            continue
        lags.append((received - parsed.ts).total_seconds())
        verdict = analyzer.evaluate(parsed)
        fix = VesselFix(
            mmsi=parsed.mmsi,
            lon=parsed.lon,
            lat=parsed.lat,
            ts=parsed.ts,
            sog_kn=parsed.sog_kn,
            cog_deg=parsed.cog_deg,
            heading_deg=parsed.heading_deg,
            name=None,
            coarse_class=None,
            source=parsed.source,
            plausibility_score=verdict.score,
            timestamp_basis=parsed.timestamp_basis,
            received_at=received,
            provider_source=parsed.provider_source,
            provider_reencoded=parsed.provider_reencoded,
        )
        if verdict.is_suspect:
            counts["suspect"] += 1
            continue
        observation = Observation(fix, received)
        references[fix.mmsi].append(observation)
        previous = latest.get(fix.mmsi)
        if previous is not None and fix.ts <= previous:
            counts["out_of_order_origins"] += 1
            continue
        if previous is not None:
            gaps.append((fix.ts - previous).total_seconds())
        latest[fix.mmsi] = fix.ts
        eligible = estimate_motion(fix, received)
        if eligible.status != "estimated":
            counts[eligible.reason] += 1
            continue
        origins.append(observation)
    counts["eligible_origins"] = len(origins)
    for values in references.values():
        values.sort(key=lambda obs: (obs.fix.ts, obs.received))

    samples: list[dict] = []
    horizons = {}
    for horizon in HORIZONS:
        skipped = Counter()
        for origin in origins:
            target = origin.received + dt.timedelta(seconds=horizon)
            initial = estimate_motion(origin.fix, target)
            if initial.status != "estimated":
                skipped[initial.reason] += 1
                continue
            reference = next(
                (
                    ref
                    for ref in references[origin.fix.mmsi]
                    if target
                    <= ref.fix.ts
                    <= target + dt.timedelta(seconds=REFERENCE_SLACK_SECONDS)
                    and ref.received > origin.received
                    and ref.fix.ts > origin.fix.ts
                ),
                None,
            )
            if reference is None:
                skipped["no_reference"] += 1
                continue
            estimate = estimate_motion(origin.fix, reference.fix.ts)
            if estimate.status != "estimated":
                skipped[estimate.reason] += 1
                continue
            _, _, error = _GEOD.inv(
                estimate.lon, estimate.lat, reference.fix.lon, reference.fix.lat
            )
            _, _, baseline = _GEOD.inv(
                origin.fix.lon, origin.fix.lat, reference.fix.lon, reference.fix.lat
            )
            course_change = (
                None
                if reference.fix.cog_deg is None
                else abs((reference.fix.cog_deg - origin.fix.cog_deg + 180) % 360 - 180)
            )
            samples.append(
                {
                    "mmsi": origin.fix.mmsi,
                    "horizon_seconds": horizon,
                    "origin_time": origin.fix.ts.isoformat(),
                    "origin_received_at": origin.received.isoformat(),
                    "reference_time": reference.fix.ts.isoformat(),
                    "reference_received_at": reference.received.isoformat(),
                    "actual_forecast_seconds": (reference.fix.ts - origin.received).total_seconds(),
                    "observation_elapsed_seconds": estimate.elapsed_seconds,
                    "receipt_lag_seconds": (origin.received - origin.fix.ts).total_seconds(),
                    "origin_sog_kn": origin.fix.sog_kn,
                    "origin_cog_deg": origin.fix.cog_deg,
                    "motion_error_m": error,
                    "hold_last_error_m": baseline,
                    "movement": "stationary" if origin.fix.sog_kn <= 0.5 else "moving",
                    "course_change_deg": course_change,
                    "course_stratum": "unknown_course"
                    if course_change is None
                    else "turning"
                    if course_change >= 10
                    else "stable",
                }
            )
        selected = [s for s in samples if s["horizon_seconds"] == horizon]
        horizons[str(horizon)] = {
            **_summary(selected),
            "origins": len(origins),
            "skipped": dict(skipped),
            "strata": {
                name: _summary(
                    [s for s in selected if name in (s["movement"], s["course_stratum"])]
                )
                for name in ("stationary", "moving", "turning", "stable", "unknown_course")
            },
        }
    positions = [obs for values in references.values() for obs in values]
    return {
        "method": "causal_constant_sog_cog_vs_hold_last",
        "reference_rule": "first provider position at/after receipt+horizon, within 15 seconds; project to its actual timestamp",
        "reference_slack_seconds": REFERENCE_SLACK_SECONDS,
        "max_observation_age_seconds": 300,
        "limitations": [
            "Later provider AIS positions are a reference, not independent location truth.",
            "Provider event time can be provider receipt time, not verified GNSS time.",
            "Repeated predictions and vessels are correlated; no confidence interval or calibrated coverage is claimed.",
            "Course-change strata use later reported course for diagnosis only, never as a prediction input.",
            "No fitting or threshold selection; horizons and 300-second cap remain fixed.",
        ],
        "input_counts": dict(counts),
        "source_counts": dict(source_counts),
        "receipt_lag_seconds": _distribution(lags),
        "update_gap_seconds": _distribution(gaps),
        "distinct_reference_vessels": len(references),
        "provider_time_range": [
            min(o.fix.ts for o in positions).isoformat(),
            max(o.fix.ts for o in positions).isoformat(),
        ]
        if positions
        else None,
        "horizons": horizons,
        "samples": samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records, malformed_lines = [], 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except ValueError:
                malformed_lines += 1
    result = evaluate_records(records)
    result["input_file"] = str(args.input)
    result["malformed_json_lines"] = malformed_lines
    result["generated_at"] = dt.datetime.now(dt.UTC).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                h: {k: v for k, v in row.items() if k != "strata"}
                for h, row in result["horizons"].items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
