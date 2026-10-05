"""Pass-level guards on GFW data: per-cell matched flags, matching artefacts, presence gaps.

GFW's SAR matching and its AIS presence feed both have measured failure modes
(PLAN §17, 2026-10-04 "Phase 6 feasibility"): one orbit's acquisitions over
Fujairah/Khor Fakkan come out fully unmatched since 2025-10-02, and on some
dates presence misses most of the ships GFW itself matched. A pass showing
either is flagged, so it is left out of the comparisons it would distort.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.labeling.gfw import _cell_key

# A pass where GFW matched none of at least this many detections while at least
# this many AIS vessels were present in the pass hour is a GFW matching artefact
# (the ~02:07 UTC orbit over Fujairah/Khor Fakkan since 2025-10-02, PLAN §17).
ARTEFACT_MIN_REFS = 10
ARTEFACT_MIN_VESSELS = 20
# Presence is incomplete on some dates: on 2026-09-09 02:14 at Fujairah it held
# 8 vessels in the box while GFW matched 64 detections to AIS there (on the busy
# pass of 2026-09-25, 140 against 69). GFW-matched ships should mostly be in
# presence (P10: 56 of 65 in the pass hour), so a pass with fewer presence
# vessels in the box than half its GFW-matched detections has a presence gap
# and is left out of the dark rate rather than counted as dark ships.
PRESENCE_GAP_RATIO = 0.5
PRESENCE_GAP_MIN_MATCHED = 10


def gfw_flag_for(lon: float, lat: float, refs: Sequence[Sequence]) -> bool | None:
    """GFW's matched flag for detections in this position's 0.01 degree cell; None if none or mixed."""
    key = _cell_key(lon, lat)
    flags = {bool(r[3]) for r in refs if _cell_key(r[0], r[1]) == key}
    return flags.pop() if len(flags) == 1 else None


def is_gfw_artefact(refs: Sequence[Sequence], *, pass_hour_vessels: int) -> bool:
    """GFW matched none of many detections although many AIS vessels were present."""
    total = sum(int(r[2]) for r in refs)
    matched = sum(int(r[2]) for r in refs if r[3])
    return total >= ARTEFACT_MIN_REFS and matched == 0 and pass_hour_vessels >= ARTEFACT_MIN_VESSELS


def is_presence_gap(*, box_vessels: int, gfw_matched: int) -> bool:
    """Presence in the box holds far fewer vessels than GFW matched to AIS there."""
    return (
        gfw_matched >= PRESENCE_GAP_MIN_MATCHED and box_vessels < PRESENCE_GAP_RATIO * gfw_matched
    )
