"""One-to-one assignment of gated SAR-AIS candidates, keeping the candidate sets.

The gate lists, per detection, every AIS vessel whose hourly track passes
within reach. Detections and vessels are then paired one-to-one with
``scipy.optimize.linear_sum_assignment``: greedy nearest-neighbour fails in
dense anchorages (PLAN §9). Pairs outside every gate get a large finite cost
and are dropped afterwards, so the solver first maximises the number of pairs
inside a gate, then minimises their total gate score. Cell-hour fusion is
often not unique - at the busy Fujairah pass of 2026-09-25 the 140 AIS vessels
of the pass hour sat in 111 cells and the median distance to another vessel was
1.0 km - so a pair is MATCHED only when neither side has another candidate.
Everything else inside a gate is AMBIGUOUS and keeps its candidate set; a
detection that loses its vessel to a better pair is ambiguous, never dark.
DARK means no candidate at all (PLAN §17, 2026-10-04 "Phase 6 feasibility").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from core.fusion.gate import Candidate

MATCHED = "matched"
AMBIGUOUS = "ambiguous"
DARK = "dark"

# Cost of a pair outside every gate: large and finite, as scipy recommends over inf.
OUTSIDE_GATE_COST = 1e6


@dataclass(frozen=True)
class Assignment:
    """One detection's outcome: its status, assigned vessel, candidate set (best first) and margin.

    ``margin`` is the runner-up's gate score minus the assigned vessel's: near
    zero for a coin toss, negative when the optimal pairing took a worse
    vessel for this detection; None without an assignment or a runner-up.
    """

    status: str
    vessel_id: str | None
    candidates: tuple[Candidate, ...]
    margin: float | None = None


def assign(found: Sequence[Sequence[Candidate]]) -> list[Assignment]:
    """Pair detections (their candidate lists, in order) with vessels one-to-one."""
    vessels = sorted({c.vessel_id for cs in found for c in cs})
    paired = _optimal_pairs(found, vessels) if vessels else {}
    claims = Counter(c.vessel_id for cs in found for c in cs)
    results: list[Assignment] = []
    for i, cs in enumerate(found):
        ordered = tuple(sorted(cs, key=lambda c: (c.score, c.vessel_id)))
        if not ordered:
            results.append(Assignment(status=DARK, vessel_id=None, candidates=()))
            continue
        vessel_id = paired.get(i)
        unique = vessel_id is not None and len(ordered) == 1 and claims[vessel_id] == 1
        results.append(
            Assignment(
                status=MATCHED if unique else AMBIGUOUS,
                vessel_id=vessel_id,
                candidates=ordered,
                margin=_margin(ordered, vessel_id),
            )
        )
    return results


def _optimal_pairs(found: Sequence[Sequence[Candidate]], vessels: list[str]) -> dict[int, str]:
    """Detection index -> vessel id for the minimum-total-score pairing inside the gates."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    column = {vessel_id: j for j, vessel_id in enumerate(vessels)}
    cost = np.full((len(found), len(vessels)), OUTSIDE_GATE_COST)
    for i, cs in enumerate(found):
        for c in cs:
            cost[i, column[c.vessel_id]] = min(cost[i, column[c.vessel_id]], c.score)
    rows, cols = linear_sum_assignment(cost)
    return {
        int(i): vessels[j]
        for i, j in zip(rows, cols, strict=True)
        if cost[i, j] < OUTSIDE_GATE_COST
    }


def _margin(ordered: tuple[Candidate, ...], vessel_id: str | None) -> float | None:
    if vessel_id is None or len(ordered) < 2:
        return None
    assigned = next(c.score for c in ordered if c.vessel_id == vessel_id)
    return min(c.score for c in ordered if c.vessel_id != vessel_id) - assigned
