"""Choosing the stored pass scripts/fuse_scene.py turns into DARK_VESSEL events.

GFW publishes AIS presence about 96 h and its SAR detections about 5 days
after a pass, so a pass younger than that cannot be fused; by default the
newest pass old enough is taken, or the one named (PLAN §17, 2026-10-04
"Phase 6 feasibility").
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.db.repository import DetectionView
from scripts.fuse_scene import pick_pass

NOW = dt.datetime(2026, 10, 5, 2, 0, tzinfo=dt.UTC)


def view(id_, scene_id, acq, name, detector="pointdet"):
    return DetectionView(
        id=id_,
        scene_id=scene_id,
        lon=56.5,
        lat=25.2,
        length_m=None,
        width_m=None,
        heading_deg=None,
        confidence=0.9,
        detector=detector,
        detector_version="kaggle-v2",
        polarisation="dv",
        pixel_count=None,
        peak_dn=None,
        acq_time=acq,
        satellite="S1C",
        product_name=name,
    )


OLD = dt.datetime(2026, 9, 10, 2, 6, tzinfo=dt.UTC)
OLDER = dt.datetime(2026, 9, 1, 2, 6, tzinfo=dt.UTC)
RECENT = dt.datetime(2026, 10, 3, 2, 14, tzinfo=dt.UTC)
VIEWS = [
    view(1, 7, RECENT, "S1C_RECENT"),
    view(2, 5, OLD, "S1D_OLD"),
    view(3, 5, OLD, "S1D_OLD"),
    view(4, 3, OLDER, "S1C_OLDER"),
]


def test_the_newest_pass_old_enough_for_gfw_is_taken():
    chosen = pick_pass(VIEWS, now=NOW)
    assert (chosen.scene_id, chosen.product_name, chosen.acquired) == (5, "S1D_OLD", OLD)
    assert [v.id for v in chosen.detections] == [2, 3]


def test_a_named_pass_is_taken_if_old_enough():
    assert pick_pass(VIEWS, now=NOW, scene="S1C_OLDER").scene_id == 3


def test_a_named_pass_too_young_for_gfw_is_refused():
    with pytest.raises(ValueError, match="days"):
        pick_pass(VIEWS, now=NOW, scene="S1C_RECENT")


def test_no_pass_old_enough_is_an_error():
    with pytest.raises(ValueError, match="no stored pass"):
        pick_pass([view(1, 7, RECENT, "S1C_RECENT")], now=NOW)


def test_model_detections_from_an_older_model_version_are_not_used():
    # The 2026-09-27 Fujairah pass in the console DB holds run 1 (kaggle-v1) detections only.
    stale = view(9, 8, dt.datetime(2026, 9, 27, 2, 14, tzinfo=dt.UTC), "S1D_STALE")
    stale = DetectionView(**{**stale.__dict__, "detector_version": "kaggle-v1"})
    cfar = view(10, 5, OLD, "S1D_OLD", detector="cfar")
    chosen = pick_pass([stale, cfar], now=NOW)
    assert chosen.product_name == "S1D_OLD"
