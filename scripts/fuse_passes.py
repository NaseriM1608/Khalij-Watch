"""Phase 6 v1: cell-hour SAR-AIS fusion over processed passes, and the dark-rate curve.

Each comparison result under --results (one JSON per area-pass from
scripts/compare_cfar_model.py) holds our frozen product model's peaks and
GFW's SAR detections with their AIS-matched flag. For every pass this script
fetches GFW hourly AIS presence for the hours H-1..H+1 over the box buffered
by about an hour of travel (cached under <out>/presence, so a rerun never
refetches), pre-filters our detections, gates them against each vessel's
hourly track at every margin of the sweep, assigns them one-to-one, and writes
one record per pass under <out>/<results folder name>/. GFW allows one 4Wings
report at a time, so passes are fetched one after another; a pass whose fetch
fails is counted and retried on the next run.

--summary pools the records into dark rates with 95% intervals from resampling
passes, overall and by region and period, and measures agreement with GFW's
matched flag (leaving out passes where a GFW matching artefact is evident).
The operating margin is chosen on the --validation folder only, never on test
(PLAN §17, 2026-10-04 "Phase 6 feasibility"; PLAN §11 Phase 6 done-when).

Run (scipy and gfwapiclient come from the uv cache; .venv-eval lacks scipy):
    uv run --offline --python 3.12 --no-project --with numpy --with scipy --with rasterio \
        --with boto3 --with sqlalchemy --with httpx --with gfw-api-python-client \
        python -u scripts/fuse_passes.py --out models/runs/kaggle-v2/fusion_v1 \
        --results models/runs/kaggle-v2/compare_val_t256 models/runs/kaggle-v2/compare_test
    ... python scripts/fuse_passes.py --summary --out models/runs/kaggle-v2/fusion_v1 \
        --validation compare_val_t256
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db.spatial import BBox  # noqa: E402
from core.fusion.assign import DARK, MATCHED, assign  # noqa: E402
from core.fusion.gate import candidates  # noqa: E402
from core.fusion.geometry import buffer_box  # noqa: E402
from core.fusion.guards import (  # noqa: E402
    artefact_by_counts,
    expected_ais,
    gap_is_judgeable,
    gfw_flag_for,
    is_gfw_artefact,
    is_presence_gap,
)
from core.fusion.prefilter import SarDetection, prefilter  # noqa: E402
from core.fusion.presence import PresenceFix, fetch_presence, hourly_tracks  # noqa: E402
from core.fusion.rate import BOOTSTRAP_RESAMPLES, dark_rate  # noqa: E402
from core.labeling.gfw import _floor_hour  # noqa: E402
from core.labeling.gfw_api import stops_the_run  # noqa: E402
from core.sar.coastline import LandPolygon  # noqa: E402
from core.sar.infrastructure import InfrastructureMask  # noqa: E402

# The frozen product model: run 2 epoch 5, 256 px tiles with 64 px overlap,
# threshold 0.60 (models/runs/kaggle-v2/val_choice.json).
PRODUCT_SHA256 = "0535c82ddaa577f1"
PRODUCT_TILE, PRODUCT_OVERLAP, PRODUCT_THRESHOLD = 256, 64, 0.60

MARGINS_M = (0.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
# About an hour of travel at lane speed: hourly cells move up to ~13 km at Fujairah (P10).
PRESENCE_BUFFER_M = 25_000.0
# GFW's dark-vessel shore convention and a box-edge margin; provisional until
# chosen on validation passes (handoff §2.1).
SHORE_M = 1_000.0
EDGE_M = 500.0
INFRASTRUCTURE_RADIUS_M = 300.0
# Upper end of Sentinel-1 IW R/V (slant range 800-900 km over 7.6 km/s, PLAN §17
# 2026-09-29). Per-sub-swath values need the annotation XML, which is not cached
# for stored passes: a bound, recorded as a shortcut in the handoff.
R_OVER_V_BOUND_S = 118.0
# Land is loaded this far around a box, so shore distances on evidence cards
# are exact up to here (the pre-filter's 1 km cut needs far less).
LAND_REACH_M = 20_000.0
# A report rarely takes over a minute; 4Wings itself gives up at about 100 s (524).
GFW_TIMEOUT_S = 180.0
# Bumped whenever the logic behind a record changes, so the summary never pools
# records made by different code. Records written before this field existed
# (the batch started 2026-10-05 01:26 UTC, commit 7b62bc7) used logic 1.
FUSION_VERSION = "1"

REGIONS = {
    "sharjah_dubai": "inner_gulf",
    "ras_al_khaimah": "inner_gulf",
    "hormuz_lanes": "hormuz",
    "fujairah": "gulf_of_oman",
    "khor_fakkan": "gulf_of_oman",
}
# (label, first day) in order; each period runs to the next one's first day.
PERIODS = (
    ("before_2026-02-28", dt.date.min),
    ("2026-02-28_to_06-29", dt.date(2026, 2, 28)),
    ("2026-06-30_to_09-07", dt.date(2026, 6, 30)),
    ("from_2026-09-08", dt.date(2026, 9, 8)),
)
TABLE_KEYS = ("dark_gfw_matched", "dark_gfw_unmatched", "ais_gfw_matched", "ais_gfw_unmatched")


def margin_key(margin_m: float) -> str:
    return f"{margin_m:g}"


def product_detections(result: dict) -> list[SarDetection]:
    """The frozen product model's detections above its threshold in one comparison result."""
    for model in result["models"].values():
        if (
            str(model.get("sha256", "")).startswith(PRODUCT_SHA256)
            and model.get("tile") == PRODUCT_TILE
            and model.get("overlap") == PRODUCT_OVERLAP
        ):
            return [
                SarDetection(lon=lon, lat=lat, score=score)
                for lon, lat, score in model["peaks"]
                if score >= PRODUCT_THRESHOLD
            ]
    raise ValueError(
        f"{result.get('scene')}: no product model ({PRODUCT_SHA256} @{PRODUCT_TILE}/{PRODUCT_OVERLAP})"
    )


def period_of(day: dt.date) -> str:
    return next(label for label, first in reversed(PERIODS) if day >= first)


def fuse_pass(
    result: dict,
    fixes: Sequence[PresenceFix],
    *,
    land: Sequence[LandPolygon],
    infrastructure: InfrastructureMask,
    box: BBox,
    margins: Sequence[float],
    r_over_v_s: float,
    shore_m: float,
    edge_m: float,
) -> dict:
    """One pass's record: pre-filter counts, tracks, and per margin the dark count and the GFW table."""
    acquired = dt.datetime.fromisoformat(result["acquired"])
    detections = product_detections(result)
    outcome = prefilter(
        detections,
        land=land,
        infrastructure=infrastructure,
        box=box,
        shore_m=shore_m,
        edge_m=edge_m,
    )
    kept = [d for d, p in zip(detections, outcome, strict=True) if p.kept]
    positions = [(d.lon, d.lat) for d in kept]
    tracks = hourly_tracks(fixes, acquired)
    hour = _floor_hour(acquired)
    pass_hour_vessels = len({f.vessel_id for f in fixes if f.hour == hour})
    box_vessels = len({f.vessel_id for f in fixes if f.hour == hour and box.contains(f.lon, f.lat)})
    gfw_refs = sum(int(r[2]) for r in result["refs"])
    gfw_matched = sum(int(r[2]) for r in result["refs"] if r[3])
    artefact = is_gfw_artefact(result["refs"], box_vessels=box_vessels)
    expected = expected_ais(refs_total=gfw_refs, matched=gfw_matched, artefact=artefact)
    flags = [gfw_flag_for(d.lon, d.lat, result["refs"]) for d in kept]
    per_margin = {}
    for margin in margins:
        assigned = assign(candidates(positions, tracks, r_over_v_s=r_over_v_s, margin_m=margin))
        per_margin[margin_key(margin)] = _margin_counts([a.status for a in assigned], flags)
    return {
        "scene": result["scene"],
        "area": result["area"],
        "region": REGIONS[result["area"]],
        "acquired": result["acquired"],
        "period": period_of(acquired.date()),
        "detections": len(detections),
        "prefilter": dict(sorted(Counter(p.reason for p in outcome if p.reason).items())),
        "kept": len(kept),
        "tracks": len(tracks),
        "stationary": sum(t.stationary for t in tracks),
        "sparse": sum(t.sparse for t in tracks),
        "pass_hour_vessels": pass_hour_vessels,
        "box_vessels": box_vessels,
        "presence_datasets": sorted({f.report_dataset for f in fixes if f.report_dataset}),
        "gfw_refs": gfw_refs,
        "gfw_matched": gfw_matched,
        # Informational: the summary recomputes both flags from the counts above.
        "gfw_artefact": artefact,
        "presence_gap": is_presence_gap(box_vessels=box_vessels, gfw_matched=expected),
        "per_margin": per_margin,
    }


def _margin_counts(statuses: Sequence[str], flags: Sequence[bool | None]) -> dict:
    """Status counts, and the 2x2 table of our call (dark or AIS) against GFW's matched flag."""
    counts = Counter(statuses)
    table = Counter(
        f"{'dark' if status == DARK else 'ais'}_gfw_{'matched' if flag else 'unmatched'}"
        for status, flag in zip(statuses, flags, strict=True)
        if flag is not None
    )
    return {
        "dark": counts[DARK],
        "matched": counts[MATCHED],
        "ambiguous": len(statuses) - counts[DARK] - counts[MATCHED],
        **{key: table[key] for key in TABLE_KEYS},
    }


def balanced_accuracy(table: dict) -> float | None:
    """Mean of: GFW-unmatched called dark, and GFW-matched called AIS; None if a class is empty."""
    unmatched = table["dark_gfw_unmatched"] + table["ais_gfw_unmatched"]
    matched = table["ais_gfw_matched"] + table["dark_gfw_matched"]
    if not unmatched or not matched:
        return None
    return (table["dark_gfw_unmatched"] / unmatched + table["ais_gfw_matched"] / matched) / 2


def acquisition_key(record: dict, index: int) -> str:
    """The overpass of a record (satellite + datatake id): boxes of one acquisition share its AIS feed."""
    parts = Path(str(record.get("scene") or "")).stem.split("_")
    return f"{parts[0]}_{parts[7]}" if len(parts) >= 8 else f"record-{index}"


def with_flags(records: Sequence[dict]) -> list[dict]:
    """Records with GFW-artefact and presence-gap flags recomputed from their stored counts.

    A presence gap belongs to the acquisition's AIS feed, not to one box, so a
    gap evident in any box of an acquisition flags all its boxes. "guarded"
    says whether the gap rule could be judged on the record itself.
    """
    flagged = []
    for i, r in enumerate(records):
        artefact = artefact_by_counts(
            refs_total=r["gfw_refs"], matched=r["gfw_matched"], box_vessels=r["box_vessels"]
        )
        expected = expected_ais(
            refs_total=r["gfw_refs"], matched=r["gfw_matched"], artefact=artefact
        )
        flagged.append(
            {
                **r,
                "acquisition": acquisition_key(r, i),
                "gfw_artefact": artefact,
                "presence_gap": is_presence_gap(box_vessels=r["box_vessels"], gfw_matched=expected),
                "guarded": gap_is_judgeable(expected),
            }
        )
    gapped = {r["acquisition"] for r in flagged if r["presence_gap"]}
    return [{**r, "presence_gap": r["acquisition"] in gapped} for r in flagged]


def _agreement(records: Sequence[dict], key: str) -> dict:
    usable = [r for r in records if not r["gfw_artefact"]]
    table = {k: sum(r["per_margin"][key][k] for r in usable) for k in TABLE_KEYS}
    return {"passes": len(usable), "balanced_accuracy": balanced_accuracy(table), **table}


def _rates(records: Sequence[dict], margins: Sequence[float], resamples: int) -> dict:
    """Pooled dark rates per margin, resampling whole acquisitions (the boxes of one overpass together)."""
    acquisitions = sorted({r["acquisition"] for r in records})
    kept = [sum(r["kept"] for r in records if r["acquisition"] == a) for a in acquisitions]
    rates = {}
    for m in margins:
        key = margin_key(m)
        dark = [
            sum(r["per_margin"][key]["dark"] for r in records if r["acquisition"] == a)
            for a in acquisitions
        ]
        rates[key] = asdict(dark_rate(dark, kept, resamples=resamples))
    return rates


def summarise(
    records: Sequence[dict], *, margins: Sequence[float], resamples: int = BOOTSTRAP_RESAMPLES
) -> dict:
    """Dark rates with intervals per margin, overall and by region and period; agreement with GFW.

    Passes with a presence gap (anywhere in their acquisition) are left out of
    both and counted, and so, separately, are kept passes whose gap could not
    be judged.
    """
    flagged = with_flags(records)
    usable = [r for r in flagged if not r["presence_gap"]]

    def by(field: str) -> dict:
        groups = sorted({r[field] for r in usable})
        return {g: _rates([r for r in usable if r[field] == g], margins, resamples) for g in groups}

    return {
        "area_passes": len(usable),
        "presence_gap_passes": len(flagged) - len(usable),
        "unguarded_passes": sum(not r["guarded"] for r in usable),
        "gfw_artefact_passes": sum(r["gfw_artefact"] for r in usable),
        "all": _rates(usable, margins, resamples),
        "by_region": by("region"),
        "by_period": by("period"),
        "gfw_agreement": {margin_key(m): _agreement(usable, margin_key(m)) for m in margins},
    }


def choose_margin(records: Sequence[dict], *, margins: Sequence[float]) -> float:
    """The margin with the highest balanced agreement with GFW's matched flag (ties: the smaller)."""
    usable = [r for r in with_flags(records) if not r["presence_gap"]]
    scored = [(balanced_accuracy(_agreement(usable, margin_key(m))), m) for m in sorted(margins)]
    scored = [(score, m) for score, m in scored if score is not None]
    if not scored:
        raise ValueError("no margin has GFW-matched and unmatched detections to agree with")
    best = max(score for score, _ in scored)
    return next(m for score, m in scored if score == best)


# ---------------------------------------------------------------------------
# Running passes (network) and writing the summary
# ---------------------------------------------------------------------------


def presence_path(out: Path, area: str, scene: str) -> Path:
    return out / "presence" / f"{area}__{Path(scene).stem}.json"


def write_json(path: Path, payload: object) -> None:
    """Write atomically, so a power-off never leaves a truncated record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_presence(path: Path) -> list[PresenceFix]:
    rows = json.loads(path.read_text(encoding="utf-8"))["fixes"]
    return [PresenceFix(**{**row, "hour": dt.datetime.fromisoformat(row["hour"])}) for row in rows]


async def _fetch(token: str, box: BBox, acquired: dt.datetime) -> list[PresenceFix]:
    import gfwapiclient

    # One client per pass: a client reused across asyncio.run loops can trip on closed connections.
    client = gfwapiclient.Client(access_token=token, timeout=GFW_TIMEOUT_S)
    return await fetch_presence(client, box, acquired)


def presence_for(out: Path, result: dict, box: BBox, token: str) -> list[PresenceFix]:
    """The pass's presence, from the cache or fetched (one 4Wings report) and cached."""
    path = presence_path(out, result["area"], result["scene"])
    query = buffer_box(box, PRESENCE_BUFFER_M)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8"))["query_box"] == asdict(
        query
    ):
        return load_presence(path)
    acquired = dt.datetime.fromisoformat(result["acquired"])
    fixes = asyncio.run(_fetch(token, query, acquired))
    write_json(
        path,
        {
            "query_box": asdict(query),
            "acquired": result["acquired"],
            "fetched_utc": dt.datetime.now(dt.UTC).isoformat(),
            "fixes": [{**asdict(f), "hour": f.hour.isoformat()} for f in fixes],
        },
    )
    return fixes


def run_passes(args: argparse.Namespace) -> int:
    from gfwapiclient.exceptions import GFWAPIClientError

    from core.config import load_settings
    from core.sar.coastline import load_land
    from core.sar.infrastructure import load_platforms
    from scripts.compare_cfar_model import AREAS, load_result
    from scripts.detect_window import box_around

    load_settings()
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        print("GFW_API_TOKEN is not set (.env)", flush=True)
        return 1
    params = {
        "margins_m": list(args.margins),
        "presence_buffer_m": PRESENCE_BUFFER_M,
        "shore_m": args.shore_m,
        "edge_m": args.edge_m,
        "infrastructure_radius_m": INFRASTRUCTURE_RADIUS_M,
        "r_over_v_s": args.r_over_v,
        "product": f"{PRODUCT_SHA256}@{PRODUCT_TILE}/{PRODUCT_OVERLAP} thr {PRODUCT_THRESHOLD}",
        "fusion_version": FUSION_VERSION,
    }
    land, platforms, failed, done = {}, {}, 0, 0
    for folder in args.results:
        for source in sorted(folder.glob("*__*.json")):
            target = args.out / folder.name / source.name
            if target.is_file():
                continue
            if args.limit and done + failed >= args.limit:
                break
            result = load_result(source)
            area = result["area"]
            box = box_around(*AREAS[area])
            if area not in land:
                land[area] = load_land(bbox=buffer_box(box, LAND_REACH_M))
                platforms[area] = load_platforms(
                    bbox=buffer_box(box, INFRASTRUCTURE_RADIUS_M + 500.0)
                )
            day = dt.datetime.fromisoformat(result["acquired"]).date()
            try:
                fixes = presence_for(args.out, result, box, token)
            except (GFWAPIClientError, OSError) as exc:
                failed += 1
                print(
                    f"  {area} {result['scene'][:40]} FAILED {type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
                if stops_the_run(exc):
                    print(
                        f"{done} passes fused; stopping: GFW may be busy, rerun later", flush=True
                    )
                    return 1
                continue
            record = fuse_pass(
                result,
                fixes,
                land=land[area],
                infrastructure=InfrastructureMask(
                    platforms[area], radius_m=INFRASTRUCTURE_RADIUS_M
                ).for_date(day),
                box=box,
                margins=args.margins,
                r_over_v_s=args.r_over_v,
                shore_m=args.shore_m,
                edge_m=args.edge_m,
            )
            write_json(
                target, {**record, "set": folder.name, "source": str(source), "params": params}
            )
            done += 1
            print(
                f"  {folder.name} {area:15s} {result['scene'][17:32]} kept {record['kept']:3d} "
                f"tracks {record['tracks']:3d} dark@0 {record['per_margin'][margin_key(args.margins[0])]['dark']:3d}"
                f"{' GFW-artefact' if record['gfw_artefact'] else ''}"
                f"{' presence-gap' if record['presence_gap'] else ''}",
                flush=True,
            )
    print(f"{done} passes fused; {failed} failed{'; rerun to retry' if failed else ''}", flush=True)
    return 1 if failed else 0


def write_summary(args: argparse.Namespace) -> int:
    sets = {}
    for folder in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name != "presence"):
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
        if records:
            sets[folder.name] = records
    for records in sets.values():
        for r in records:  # records from before the field existed ran fusion logic 1
            r["params"].setdefault("fusion_version", "1")
    params = {json.dumps(r["params"], sort_keys=True) for records in sets.values() for r in records}
    if len(params) != 1:
        print(
            f"records were made with {len(params)} different settings; refusing to pool them",
            flush=True,
        )
        return 1
    margins = json.loads(params.pop())["margins_m"]
    if args.validation not in sets:
        print(f"no validation records in {args.out / args.validation}", flush=True)
        return 1
    for name, records in sets.items():
        sources = len(list(Path(records[0]["source"]).parent.glob("*__*.json")))
        if len(records) < sources:
            print(
                f"WARNING: {name} holds {len(records)} of {sources} passes; the batch is incomplete"
            )
    chosen = choose_margin(sets[args.validation], margins=margins)
    summary = {
        "operating_margin_m": chosen,
        "chosen_on": args.validation,
        "sets": {
            name: summarise(records, margins=margins, resamples=args.resamples)
            for name, records in sets.items()
        },
    }
    write_json(args.out / "summary.json", summary)
    key = margin_key(chosen)
    print(f"operating margin {chosen:g} m (chosen on {args.validation})", flush=True)
    for name, s in summary["sets"].items():
        rate, agreement = s["all"][key], s["gfw_agreement"][key]
        print(
            f"  {name:22s} area-passes {s['area_passes']:3d} (gap {s['presence_gap_passes']}, "
            f"unguarded {s['unguarded_passes']}) dark {rate['dark']}/{rate['kept']} = {rate['rate']} "
            f"[{rate['low']}, {rate['high']}]  GFW balanced accuracy {agreement['balanced_accuracy']}",
            flush=True,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--validation", default="compare_val_t256", help="results folder the margin is chosen on"
    )
    parser.add_argument("--margins", type=float, nargs="+", default=list(MARGINS_M))
    parser.add_argument("--shore-m", type=float, default=SHORE_M)
    parser.add_argument("--edge-m", type=float, default=EDGE_M)
    parser.add_argument("--r-over-v", type=float, default=R_OVER_V_BOUND_S)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--limit", type=int, default=0, help="fuse at most this many new passes (0 = all)"
    )
    args = parser.parse_args(argv)
    if args.summary:
        return write_summary(args)
    if not args.results:
        parser.error("--results is required unless --summary")
    return run_passes(args)


if __name__ == "__main__":
    sys.exit(main())
