"""Tests for the combined dataset index over area builds."""

from __future__ import annotations

from core.labeling.index import (
    CENTRE_NEGATIVE,
    FULL_NEGATIVE,
    IGNORE,
    POSITIVE,
    as_persistent,
    chip_objects,
    index_record,
    persistent_spots,
    supervision_for,
)


def rec(
    chip_id: str,
    *,
    kind: str = "vessel",
    source: str = "gfw",
    scene: str = "S.SAFE",
    row: float = 1000.0,
    col: float = 2000.0,
    lon: float = 56.5,
    lat: float = 25.2,
) -> dict:
    return {
        "chip_id": chip_id,
        "scene": scene,
        "split": "train",
        "shard": f"shards/{scene}.npz",
        "index": 0,
        "lon": lon,
        "lat": lat,
        "row": row,
        "col": col,
        "kind": kind,
        "source": source,
        "mmsi": None,
        "length_m": None,
        "vessel_type": None,
        "extra": {},
    }


# --- supervision ---------------------------------------------------------------


def test_supervision_follows_kind_and_negative_source():
    assert supervision_for(rec("a", kind="vessel")) == POSITIVE
    assert supervision_for(rec("a", kind="unlabelled")) == IGNORE
    assert supervision_for(rec("a", kind="negative", source="sea")) == FULL_NEGATIVE
    assert supervision_for(rec("a", kind="negative", source="clutter")) == FULL_NEGATIVE
    for source in ("platform", "coast", "persistent"):
        assert supervision_for(rec("a", kind="negative", source=source)) == CENTRE_NEGATIVE


def test_a_detection_gfw_saw_without_ais_is_a_positive():
    """kaggle-v1 ignored these and learned to miss non-AIS vessels (recall 0.49 vs CFAR 0.78)."""
    seen = rec("a", kind="unlabelled")
    seen["extra"] = {"gfw_status": "gfw_unmatched"}
    absent = rec("b", kind="unlabelled")
    absent["extra"] = {"gfw_status": "gfw_absent"}

    assert supervision_for(seen) == POSITIVE
    assert supervision_for(absent) == IGNORE


def test_gfw_unmatched_neighbours_appear_as_vessels_inside_other_chips():
    centre = rec("S/d0000")
    neighbour = rec("S/d0001", kind="unlabelled", row=1010.0)
    neighbour["extra"] = {"gfw_status": "gfw_unmatched"}

    assert chip_objects(centre, [centre, neighbour]) == [[10.0, 0.0, "vessel"]]


# --- namespacing -----------------------------------------------------------------


def test_index_record_prefixes_ids_and_shards_with_the_area():
    """One pass covering two boxes gives both areas chip "S/d0000"."""
    out = index_record(rec("S/d0000"), "khor_fakkan", objects=[])

    assert out["chip_id"] == "khor_fakkan/S/d0000"
    assert out["shard"] == "khor_fakkan/shards/S.SAFE.npz"
    assert out["area"] == "khor_fakkan"
    assert out["supervision"] == POSITIVE
    assert out["objects"] == []


def test_index_record_does_not_change_its_input():
    original = rec("S/d0000")
    snapshot = dict(original)

    index_record(original, "fujairah", objects=[])

    assert original == snapshot


# --- neighbours inside a chip ---------------------------------------------------------


def test_chip_objects_lists_other_detections_inside_the_chip():
    centre = rec("S/d0000", row=1000.0, col=2000.0)
    near = rec("S/d0001", kind="unlabelled", row=1010.0, col=1970.0)
    far = rec("S/d0002", row=1100.0, col=2000.0)

    objects = chip_objects(centre, [centre, near, far])

    assert objects == [[10.0, -30.0, "unlabelled"]]


def test_chip_objects_ignore_other_scenes_and_negatives():
    centre = rec("S/d0000")
    other_scene = rec("T/d0000", scene="T.SAFE")
    negative = rec("S/n0000", kind="negative", source="sea", row=1005.0)

    assert chip_objects(centre, [centre, other_scene, negative]) == []


def test_chip_objects_of_a_negative_list_the_detections_around_it():
    negative = rec("S/hc0000", kind="negative", source="coast")
    vessel = rec("S/d0003", row=1020.0, col=2040.0)

    assert chip_objects(negative, [vessel]) == [[20.0, 40.0, "vessel"]]


# --- persistent structures -----------------------------------------------------------


def test_a_spot_detected_in_most_scenes_and_never_labelled_is_persistent():
    scenes = [f"S{i}.SAFE" for i in range(10)]
    structure = [
        rec(f"S{i}/d0000", kind="unlabelled", scene=s, lon=56.40001 + i * 1e-5, lat=25.3)
        for i, s in enumerate(scenes[:6])
    ]
    moving = [
        rec(f"S{i}/d0001", scene=s, lon=56.1 + i * 0.01, lat=25.1) for i, s in enumerate(scenes)
    ]

    spots = persistent_spots(structure + moving, scene_count=len(scenes))

    assert len(spots) == 1
    assert all(spots.contains(r) for r in structure)
    assert not any(spots.contains(r) for r in moving)


def test_a_labelled_vessel_at_the_spot_means_it_is_not_a_structure():
    """A ship anchored for weeks is persistent too; AIS says it is a ship."""
    scenes = [f"S{i}.SAFE" for i in range(10)]
    spot = [
        rec(f"S{i}/d0000", kind="unlabelled", scene=s, lon=56.4, lat=25.3)
        for i, s in enumerate(scenes[:6])
    ]
    spot[0] = rec("S0/d0000", kind="vessel", scene=scenes[0], lon=56.4, lat=25.3)

    assert len(persistent_spots(spot, scene_count=len(scenes))) == 0


def test_a_persistent_detection_becomes_a_centre_negative_and_remembers_its_kind():
    original = rec("S/d0000", kind="unlabelled")
    original["extra"] = {"gfw_status": "gfw_absent"}

    relabelled = as_persistent(original)

    assert relabelled["kind"] == "negative"
    assert relabelled["source"] == "persistent"
    assert relabelled["extra"] == {"gfw_status": "gfw_absent", "relabelled_from": "unlabelled"}
    assert supervision_for(relabelled) == CENTRE_NEGATIVE
    assert original["kind"] == "unlabelled"
    assert original["extra"] == {"gfw_status": "gfw_absent"}


def test_chip_objects_show_persistent_structures_by_name():
    centre = rec("S/d0000")
    structure = as_persistent(rec("S/d0009", kind="unlabelled", row=1030.0))

    assert chip_objects(centre, [centre, structure]) == [[30.0, 0.0, "persistent"]]


def test_an_occasional_spot_is_not_persistent():
    scenes = [f"S{i}.SAFE" for i in range(10)]
    spot = [
        rec(f"S{i}/d0000", kind="unlabelled", scene=s, lon=56.4, lat=25.3)
        for i, s in enumerate(scenes[:2])
    ]

    assert len(persistent_spots(spot, scene_count=len(scenes))) == 0
