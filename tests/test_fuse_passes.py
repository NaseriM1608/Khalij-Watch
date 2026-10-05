"""Phase 6 v1 batch fusion over processed passes (scripts/fuse_passes.py), no network.

Each comparison result (models/runs/kaggle-v2/compare_*) holds one area-pass:
our product model's peaks, CFAR and GFW's SAR detections with their matched
flag. The runner adds GFW hourly AIS presence, pre-filters our detections,
gates and assigns them, and records per pass the counts the dark-rate curve
and the agreement with GFW's matched flag need (PLAN §17, 2026-10-04 "Phase 6
feasibility").
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.db.spatial import BBox
from core.fusion.presence import PresenceFix
from core.sar.infrastructure import InfrastructureMask
from scripts.fuse_passes import (
    choose_margin,
    fuse_pass,
    gfw_flag_for,
    is_gfw_artefact,
    is_presence_gap,
    period_of,
    product_detections,
    stops_the_run,
    summarise,
)

ACQUIRED = "2026-09-25T14:15:51.146302+00:00"
BOX = BBox(56.43, 25.13, 56.63, 25.31)
PRODUCT = {"sha256": "0535c82ddaa577f1", "tile": 256, "overlap": 64}


def result(peaks, refs=(), area="fujairah", acquired=ACQUIRED):
    return {
        "scene": "S1C_TEST.SAFE",
        "area": area,
        "acquired": acquired,
        "refs": [list(r) for r in refs],
        "gfw_in_box": sum(r[2] for r in refs),
        "models": {
            "v1": {
                "sha256": "8910aba9e98c1bc2",
                "tile": 128,
                "overlap": 32,
                "peaks": [[56.0, 25.0, 0.99]],
            },
            "run2": {**PRODUCT, "peaks": [list(p) for p in peaks]},
        },
    }


def anchored(vessel_id, lon, lat):
    return [
        PresenceFix(
            vessel_id=vessel_id, hour=dt.datetime(2026, 9, 25, h, tzinfo=dt.UTC), lon=lon, lat=lat
        )
        for h in (13, 14, 15)
    ]


def fuse(res, fixes, margins=(0.0, 1000.0)):
    return fuse_pass(
        res,
        fixes,
        land=[],
        infrastructure=InfrastructureMask([], radius_m=300.0),
        box=BOX,
        margins=margins,
        r_over_v_s=118.0,
        shore_m=1000.0,
        edge_m=500.0,
    )


# ---------------------------------------------------------------------------
# Inputs from a comparison result
# ---------------------------------------------------------------------------


def test_only_the_frozen_product_model_above_its_threshold_is_used():
    detections = product_detections(result([(56.50, 25.20, 0.95), (56.52, 25.22, 0.59)]))
    assert [(d.lon, d.lat, d.score) for d in detections] == [(56.50, 25.20, 0.95)]


def test_a_result_without_the_product_model_is_rejected():
    res = result([])
    del res["models"]["run2"]
    with pytest.raises(ValueError, match="product model"):
        product_detections(res)


def test_gfw_flag_is_the_matched_flag_of_gfw_detections_in_the_same_cell():
    refs = [
        (56.50, 25.20, 1, True),
        (56.53, 25.20, 2, False),
        (56.55, 25.20, 1, True),
        (56.55, 25.20, 1, False),
    ]
    assert gfw_flag_for(56.502, 25.199, refs) is True
    assert gfw_flag_for(56.531, 25.201, refs) is False
    assert gfw_flag_for(56.551, 25.20, refs) is None  # mixed cell
    assert gfw_flag_for(56.60, 25.30, refs) is None  # no GFW detection


def test_a_pass_where_gfw_matched_nothing_amid_many_ais_vessels_is_an_artefact():
    refs = [(56.50, 25.20, 12, False)]
    assert is_gfw_artefact(refs, box_vessels=150)
    assert not is_gfw_artefact(refs, box_vessels=3)  # a quiet sea: unmatched is plausible
    assert not is_gfw_artefact([(56.50, 25.20, 12, False), (56.51, 25.2, 1, True)], box_vessels=150)
    assert not is_gfw_artefact([(56.50, 25.20, 2, False)], box_vessels=150)  # too few to judge


def test_a_pass_whose_presence_misses_most_of_gfws_matched_ships_has_a_presence_gap():
    # 2026-09-09 02:14 Fujairah: 8 presence vessels in the box against 64 GFW-matched detections.
    assert is_presence_gap(box_vessels=8, gfw_matched=64)
    assert not is_presence_gap(box_vessels=140, gfw_matched=69)  # the busy pass of 2026-09-25
    assert not is_presence_gap(box_vessels=1, gfw_matched=1)  # too few to judge


def test_periods_follow_the_plan_strata():
    assert period_of(dt.date(2025, 10, 7)) == "before_2026-02-28"
    assert period_of(dt.date(2026, 2, 28)) == "2026-02-28_to_06-29"
    assert period_of(dt.date(2026, 7, 15)) == "2026-06-30_to_09-07"
    assert period_of(dt.date(2026, 9, 8)) == "from_2026-09-08"


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------


def test_a_pass_record_counts_dark_and_ais_detections_per_margin():
    res = result(
        peaks=[
            (56.502, 25.201, 0.9),
            (56.58, 25.27, 0.9),
        ],  # one at an anchored vessel, one 1.2 km off another
        refs=[(56.50, 25.20, 1, True), (56.58, 25.27, 1, False)],
    )
    fixes = anchored("a", 56.50, 25.20) + anchored("b", 56.58, 25.27 + 1200 / 110_540)
    record = fuse(res, fixes)
    assert (record["detections"], record["kept"], record["tracks"], record["stationary"]) == (
        2,
        2,
        2,
        2,
    )
    assert record["pass_hour_vessels"] == 2
    tight, wide = record["per_margin"]["0"], record["per_margin"]["1000"]
    assert (tight["dark"], tight["matched"] + tight["ambiguous"]) == (1, 1)
    assert wide["dark"] == 0
    # Against GFW at margin 0: the anchored match agrees; the dark one agrees with GFW's unmatched.
    assert (tight["ais_gfw_matched"], tight["dark_gfw_unmatched"]) == (1, 1)
    assert (wide["ais_gfw_matched"], wide["ais_gfw_unmatched"]) == (1, 1)


def test_pre_filtered_detections_are_counted_by_reason_and_left_out():
    res = result(peaks=[(56.502, 25.201, 0.9), (56.5025, 25.2012, 0.8), (56.432, 25.20, 0.9)])
    record = fuse(res, anchored("a", 56.50, 25.20))
    assert record["prefilter"] == {"duplicate": 1, "edge": 1}
    assert record["kept"] == 1


def test_a_pass_record_carries_region_period_and_gfw_totals():
    res = result(
        peaks=[], refs=[(56.50, 25.20, 3, True), (56.51, 25.20, 1, False)], area="khor_fakkan"
    )
    record = fuse(res, [])
    assert (record["region"], record["period"]) == ("gulf_of_oman", "from_2026-09-08")
    assert (record["gfw_refs"], record["gfw_matched"], record["gfw_artefact"]) == (4, 3, False)


def test_a_pass_record_counts_presence_vessels_inside_the_box_and_flags_a_gap():
    refs = [(56.50 + 0.01 * i, 25.20, 1, True) for i in range(12)]
    outside = anchored("far", 56.80, 25.40)  # in the buffered query, outside the 20 km box
    record = fuse(result(peaks=[], refs=refs), anchored("a", 56.50, 25.20) + outside)
    assert (record["pass_hour_vessels"], record["box_vessels"]) == (2, 1)
    assert record["presence_gap"]


# ---------------------------------------------------------------------------
# Summary and the operating margin
# ---------------------------------------------------------------------------


def rec(
    dark,
    kept,
    *,
    region="gulf_of_oman",
    period="before_2026-02-28",
    artefact=False,
    gap=False,
    table=(0, 0, 0, 0),
    scene=None,
    refs=20,
    matched=None,
    box_vessels=None,
):
    """A pass record described by the stored counts the summary derives its flags from."""
    dgm, dgu, agm, agu = table
    if matched is None:
        matched = 0 if artefact else refs
    if box_vessels is None:
        box_vessels = 0 if gap else 50
    return {
        "scene": scene,
        "region": region,
        "period": period,
        "kept": kept,
        "gfw_refs": refs,
        "gfw_matched": matched,
        "box_vessels": box_vessels,
        "per_margin": {
            "0": {
                "dark": dark,
                "dark_gfw_matched": dgm,
                "dark_gfw_unmatched": dgu,
                "ais_gfw_matched": agm,
                "ais_gfw_unmatched": agu,
            },
        },
    }


def test_the_summary_gives_a_dark_rate_with_interval_per_margin_overall_and_by_stratum():
    records = [rec(1, 10), rec(3, 10, region="inner_gulf")]
    summary = summarise(records, margins=(0.0,), resamples=200)
    assert summary["all"]["0"]["rate"] == pytest.approx(0.2)
    assert summary["by_region"]["inner_gulf"]["0"]["rate"] == pytest.approx(0.3)
    assert summary["by_period"]["before_2026-02-28"]["0"]["passes"] == 2


def test_agreement_with_gfw_leaves_out_artefact_passes():
    records = [rec(0, 10, table=(0, 2, 8, 0)), rec(0, 10, artefact=True, table=(0, 0, 0, 10))]
    agreement = summarise(records, margins=(0.0,), resamples=10)["gfw_agreement"]["0"]
    assert agreement["passes"] == 1
    assert agreement["balanced_accuracy"] == pytest.approx(1.0)


def test_the_operating_margin_maximises_balanced_agreement_with_gfw():
    def two_margins(t0, t1):
        r = rec(0, 10, table=t0)
        r["per_margin"]["500"] = dict(r["per_margin"]["0"])
        r["per_margin"]["500"].update(
            zip(
                ("dark_gfw_matched", "dark_gfw_unmatched", "ais_gfw_matched", "ais_gfw_unmatched"),
                t1,
                strict=True,
            )
        )
        return r

    # At 0 m GFW-matched ships look dark (gate too tight); at 500 m both classes agree better.
    records = [two_margins((4, 2, 4, 0), (0, 2, 8, 0))]
    assert choose_margin(records, margins=(0.0, 500.0)) == 500.0


def test_presence_gap_passes_are_left_out_of_rates_and_agreement_and_counted():
    records = [rec(1, 10, table=(0, 1, 9, 0)), rec(9, 10, gap=True, table=(9, 0, 1, 0))]
    summary = summarise(records, margins=(0.0,), resamples=10)
    assert summary["presence_gap_passes"] == 1
    assert summary["all"]["0"]["rate"] == pytest.approx(0.1)
    assert summary["gfw_agreement"]["0"]["passes"] == 1


def test_a_pass_record_keeps_the_presence_dataset_versions_served():
    fixes = [
        PresenceFix(
            vessel_id="a",
            hour=dt.datetime(2026, 9, 25, 14, tzinfo=dt.UTC),
            lon=56.50,
            lat=25.20,
            report_dataset="public-global-presence:v4.0",
        )
    ]
    record = fuse(result(peaks=[]), fixes)
    assert record["presence_datasets"] == ["public-global-presence:v4.0"]


# ---------------------------------------------------------------------------
# Review fixes (2026-10-05): flags from counts, per acquisition; resampling
# acquisitions; which GFW errors stop the run
# ---------------------------------------------------------------------------

FUJ = "S1C_IW_GRDH_1SDV_20260909T021440_20260909T021505_009364_012A07_08A2_COG.SAFE"
RAK = (
    "S1C_IW_GRDH_1SDV_20260909T021415_20260909T021440_009364_012A07_1F2C_COG.SAFE"  # same datatake
)
OTHER = "S1D_IW_GRDH_1SDV_20260910T020640_20260910T020705_004509_00860C_1202_COG.SAFE"


def test_a_presence_gap_found_in_one_box_covers_every_box_of_that_acquisition():
    records = [
        rec(9, 10, scene=FUJ, refs=64, box_vessels=8),  # gap evident here
        rec(
            5, 10, scene=RAK, refs=7, box_vessels=1, region="inner_gulf"
        ),  # too few GFW to judge alone
        rec(1, 10, scene=OTHER),
    ]
    summary = summarise(records, margins=(0.0,), resamples=10)
    assert summary["presence_gap_passes"] == 2
    assert summary["all"]["0"]["rate"] == pytest.approx(0.1)


def test_an_artefact_pass_is_judged_for_a_gap_against_all_gfw_detections():
    # GFW matched none of 60 by fault (25 AIS vessels in the box show it is an
    # artefact), but 25 is under half of the 60 ships that should carry AIS: a gap.
    gapped = rec(8, 10, scene=OTHER, refs=60, artefact=True, box_vessels=25)
    summary = summarise([gapped, rec(1, 10, scene=FUJ)], margins=(0.0,), resamples=10)
    assert summary["presence_gap_passes"] == 1


def test_no_matches_amid_few_ais_vessels_cannot_be_judged_and_is_counted_unguarded():
    # 40 GFW detections, none matched, 5 presence vessels: an artefact with a gap,
    # or 35 dark ships - the counts cannot tell, so the pass stays in, unguarded.
    unclear = rec(8, 10, scene=OTHER, refs=40, matched=0, box_vessels=5)
    summary = summarise([unclear, rec(1, 10, scene=FUJ)], margins=(0.0,), resamples=10)
    assert (summary["presence_gap_passes"], summary["unguarded_passes"]) == (0, 1)


def test_passes_the_gap_guard_cannot_judge_are_counted():
    summary = summarise(
        [rec(1, 10, scene=OTHER, refs=3), rec(1, 10, scene=FUJ)], margins=(0.0,), resamples=10
    )
    assert summary["unguarded_passes"] == 1


def test_boxes_of_one_acquisition_are_resampled_together():
    summary = summarise(
        [rec(1, 10, scene=FUJ), rec(3, 10, scene=RAK), rec(0, 10, scene=OTHER)],
        margins=(0.0,),
        resamples=50,
    )
    assert summary["all"]["0"]["passes"] == 2  # two acquisitions
    assert summary["area_passes"] == 3


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(status_code)
        self.status_code = status_code


@pytest.mark.parametrize(
    ("exc", "stops"),
    [
        (_StatusError(422), False),  # this pass's request is wrong: skip it
        (_StatusError(404), False),
        (_StatusError(429), True),  # GFW's one-report lock is held
        (_StatusError(401), True),  # the token is wrong for every pass
        (_StatusError(403), True),
        (_StatusError(524), True),  # a report may still be computing
        (_StatusError(503), True),
        (TimeoutError("read timed out"), True),
    ],
)
def test_only_per_pass_request_errors_let_the_run_continue(exc, stops):
    assert stops_the_run(exc) is stops
