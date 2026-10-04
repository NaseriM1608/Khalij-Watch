"""One-to-one assignment of gated candidates, with candidate sets kept.

The gate (core.fusion.gate) lists, per detection, every AIS vessel whose
hourly track passes close enough. Assignment then pairs detections and
vessels one-to-one by minimum total gate score (scipy's linear_sum_assignment;
PLAN §9 warns that greedy nearest-neighbour fails in dense anchorages). In
cell-hour fusion many pairs are not unique, so each result keeps its whole
candidate set, and a detection that loses the assignment is ambiguous, not
dark (PLAN §17, 2026-10-04 "Phase 6 feasibility").
"""

from __future__ import annotations

import pytest

from core.fusion.assign import AMBIGUOUS, DARK, MATCHED, assign
from core.fusion.gate import Candidate

GATE = 1000.0


def cand(vessel_id: str, score: float) -> Candidate:
    return Candidate(vessel_id=vessel_id, distance_m=score * GATE, gate_m=GATE)


def test_a_detection_without_candidates_is_dark():
    (result,) = assign([[]])
    assert result.status == DARK
    assert result.vessel_id is None
    assert result.candidates == ()


def test_a_lone_pair_is_matched():
    (result,) = assign([[cand("v1", 0.2)]])
    assert result.status == MATCHED
    assert result.vessel_id == "v1"
    assert result.margin is None


def test_pairs_with_competing_candidates_are_ambiguous_but_assigned():
    first, second = assign([[cand("v1", 0.1), cand("v2", 0.6)], [cand("v2", 0.2), cand("v1", 0.5)]])
    assert (first.status, first.vessel_id) == (AMBIGUOUS, "v1")
    assert (second.status, second.vessel_id) == (AMBIGUOUS, "v2")
    assert first.margin == pytest.approx(0.5)  # the runner-up's score minus the assigned one's


def test_the_optimal_assignment_beats_greedy_nearest():
    # Greedy would give d1 its nearest (v1) and leave d2 with nothing.
    d1, d2 = assign([[cand("v1", 0.30), cand("v2", 0.40)], [cand("v1", 0.35)]])
    assert d1.vessel_id == "v2"
    assert d2.vessel_id == "v1"


def test_a_detection_that_loses_its_only_vessel_is_ambiguous_not_dark():
    d1, d2 = assign([[cand("v1", 0.1)], [cand("v1", 0.4)]])
    assert (d1.status, d1.vessel_id) == (AMBIGUOUS, "v1")
    assert (d2.status, d2.vessel_id) == (AMBIGUOUS, None)
    assert [c.vessel_id for c in d2.candidates] == ["v1"]


def test_a_unique_pair_stays_matched_beside_a_crowded_one():
    results = assign([[cand("a", 0.3)], [cand("b", 0.1), cand("c", 0.2)], [cand("c", 0.1)]])
    assert [r.status for r in results] == [MATCHED, AMBIGUOUS, AMBIGUOUS]
    assert [r.vessel_id for r in results] == ["a", "b", "c"]


def test_results_keep_the_input_order_and_candidate_sets():
    results = assign([[], [cand("v1", 0.2), cand("v2", 0.9)], []])
    assert [r.status for r in results] == [DARK, AMBIGUOUS, DARK]
    assert [c.vessel_id for c in results[1].candidates] == ["v1", "v2"]


def test_no_detections_give_no_results():
    assert assign([]) == []
