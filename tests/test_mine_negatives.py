"""Tests for the hard negative mining script's bookkeeping."""

from __future__ import annotations

from core.labeling.manifest import ChipRecord, records_from_jsonl
from scripts.mine_negatives import (
    SOURCE_PREFIX,
    done_scenes,
    hard_requests,
    platform_window_points,
    write_scene_records,
)


def record(scene: str, chip_id: str) -> ChipRecord:
    return ChipRecord(
        chip_id=chip_id,
        scene=scene,
        split="train",
        shard="hardneg/shards/x.npz",
        index=0,
        lon=56.5,
        lat=25.2,
        row=10.0,
        col=20.0,
        kind="negative",
        source="coast",
    )


def test_written_scene_records_read_back_and_leave_no_temporary_file(tmp_path):
    records = [record("A.SAFE", "A/hc0000"), record("A.SAFE", "A/hc0001")]

    path = write_scene_records(tmp_path, "A", records)

    assert records_from_jsonl(path.read_text(encoding="utf-8")) == records
    assert [p.name for p in tmp_path.iterdir()] == ["A.jsonl"]


def test_done_scenes_are_those_with_a_record_file(tmp_path):
    write_scene_records(tmp_path, "A", [record("A.SAFE", "A/hc0000")])
    write_scene_records(tmp_path, "B", [])
    (tmp_path / "C.jsonl.tmp").write_text("{partial", encoding="utf-8")

    assert done_scenes(tmp_path) == {"A", "B"}


def test_done_scenes_of_a_missing_directory_is_empty(tmp_path):
    assert done_scenes(tmp_path / "absent") == set()


def test_hard_requests_name_chips_by_source_in_scene_coordinates():
    groups = {"platform": [(100.0, 200.0)], "clutter": [(5.0, 6.0), (7.0, 8.0)]}

    requests = hard_requests("S", groups, window_offset=(1000, 2000))

    assert [(r.chip_id, r.row, r.col, source) for r, source in requests] == [
        (f"S/{SOURCE_PREFIX['platform']}0000", 1100.0, 2200.0, "platform"),
        (f"S/{SOURCE_PREFIX['clutter']}0000", 1005.0, 2006.0, "clutter"),
        (f"S/{SOURCE_PREFIX['clutter']}0001", 1007.0, 2008.0, "clutter"),
    ]


def test_chip_ids_never_clash_with_detections_or_sea_negatives():
    prefixes = set(SOURCE_PREFIX.values())

    assert len(prefixes) == len(SOURCE_PREFIX)
    assert not prefixes & {"d", "n"}


def test_platform_points_move_from_pixel_corners_to_window_centres():
    """raster_coords is corner-based; chips are cut around pixel centres."""
    points = platform_window_points([1010.5], [2020.5], window_offset=(1000, 2000))

    assert points == [(10.0, 20.0)]
