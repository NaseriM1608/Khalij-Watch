"""Tests for the dataset build's per-scene counts and summary.json rows."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from core.labeling.gfw import AMBIGUOUS, GFW_ABSENT, GFW_UNMATCHED, LABELLED, GfwCell
from core.labeling.manifest import ChipRecord
from scripts.build_dataset import SceneResult, scene_counts, summary_path, summary_rows

HOUR = dt.datetime(2025, 9, 17, 14, tzinfo=dt.UTC)


def record(kind: str, status: str | None = None, reason: str | None = None) -> ChipRecord:
    extra = {}
    if status:
        extra["gfw_status"] = status
    if reason:
        extra["gfw_reason"] = reason
    return ChipRecord(
        chip_id="scene/d0000",
        scene="scene.SAFE",
        split="train",
        shard="shards/scene.npz",
        index=0,
        lon=56.5,
        lat=25.2,
        row=0.0,
        col=0.0,
        kind=kind,
        source="gfw" if kind == "vessel" else "sea",
        extra=extra,
    )


def cell(matched: bool, detections: int = 1) -> GfwCell:
    return GfwCell(lat=25.2, lon=56.5, hour=HOUR, detections=detections, matched=matched)


def processed(chips_skipped: int) -> SceneResult:
    result = SceneResult(scene="scene.SAFE", acquired=HOUR.isoformat())
    result.counts = scene_counts(
        detections=3, records=[record("negative")], chips_skipped=chips_skipped, pass_cells=[]
    )
    return result


def test_scene_counts_tally_statuses_reasons_negatives_and_gfw_cells():
    records = [
        record("vessel", LABELLED),
        record("vessel", LABELLED),
        record("unlabelled", GFW_UNMATCHED),
        record("unlabelled", GFW_ABSENT),
        record("unlabelled", AMBIGUOUS, "edge"),
        record("unlabelled", AMBIGUOUS, "crowded_ours"),
        record("unlabelled", AMBIGUOUS, "crowded_ours"),
        record("unlabelled", AMBIGUOUS, "crowded_gfw"),
        record("negative"),
    ]
    cells = [cell(True), cell(True), cell(False, detections=3)]

    counts = scene_counts(detections=9, records=records, chips_skipped=1, pass_cells=cells)

    assert counts == {
        "detections": 9,
        "labelled": 2,
        "gfw_unmatched": 1,
        "gfw_absent": 1,
        "ambiguous": 4,
        "ambiguous_edge": 1,
        "ambiguous_crowded_ours": 2,
        "ambiguous_crowded_gfw": 1,
        "negatives": 1,
        "chips_skipped": 1,
        "gfw_matched_cells": 2,
        "gfw_unmatched_detections": 3,
    }


def test_summary_row_of_a_processed_scene_keeps_its_skip_reason_empty():
    """The chip-skip count once shared the "skipped" key with the scene's skip
    reason and overwrote it, so processed scenes looked skipped in summary.json."""
    row = summary_rows([processed(chips_skipped=2)])[0]

    assert row["skipped"] is None
    assert row["chips_skipped"] == 2


def test_summary_row_of_a_skipped_scene_carries_the_reason():
    result = SceneResult(
        scene="skipped.SAFE", acquired=HOUR.isoformat(), skipped="no GFW detections"
    )

    assert summary_rows([result]) == [
        {
            "scene": "skipped.SAFE",
            "acquired": HOUR.isoformat(),
            "error": None,
            "skipped": "no GFW detections",
        }
    ]


def test_summary_path_names_the_build_period():
    """Builds of different periods share an output directory and its manifest;
    one fixed summary name let each build erase the previous period's summary."""
    path = summary_path(Path("out"), dt.date(2024, 1, 1), dt.date(2025, 1, 1))

    assert path == Path("out") / "summary_2024-01-01_2025-01-01.json"


def test_summary_paths_of_different_periods_differ():
    first = summary_path(Path("out"), dt.date(2024, 1, 1), dt.date(2025, 1, 1))
    second = summary_path(Path("out"), dt.date(2025, 1, 1), dt.date(2026, 1, 1))

    assert first != second
