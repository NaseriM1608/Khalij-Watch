"""Measure CFAR false alarms on real sea clutter.

Real imagery has no truth map: nobody knows which pixels are vessels. What can
be measured honestly is how often the CFAR test fires on sea clutter away from
every object it found. On ideal Gamma clutter that rate equals the requested
false-alarm probability, and the synthetic tests confirm it does. Real sea is
heavier-tailed than the Gamma model, so any excess over the requested value is
model error that the detector's Pfa setting does not show.

How it works
------------
1. Run CFAR once at a reference setting and keep what survives morphological
   opening: vessels, structures the platform mask missed, anything solid
   enough to be a target.
2. Grow those objects by ``exclusion_px`` so their edges and near sidelobes
   are excluded too. What remains, less masked pixels, is the clutter area.
3. For each requested Pfa, count CFAR hits inside that fixed clutter area.
   The area is the same for every setting, so the rows are comparable.

This is a PIXEL false-alarm rate. Object-level false alarms per 100 km2 need
AIS truth to tell vessels from clutter, and wait for Phase 4. The candidate
count reported with each row is an upper bound on them, not a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from core.sar.cfar import CfarConfig, cfar_detect
from core.sar.detect import _binary_opening, detect_candidates
from core.sar.morphology import dilate

# From loose to beyond the operating point. Loose settings give enough hits to
# measure a rate precisely; strict ones show behaviour at the operating point.
DEFAULT_SWEEP = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

# 25 px is 250 m: beyond the hull and near sidelobes of all but the brightest
# vessels. Azimuth ambiguities sit kilometres away and are counted, correctly,
# as false alarms.
DEFAULT_EXCLUSION_PX = 25


@dataclass(frozen=True)
class PfaMeasurement:
    """CFAR behaviour on sea clutter at one false-alarm setting."""

    requested_pfa: float
    alpha: float
    clutter_pixels: int
    clutter_hits: int
    candidates: int

    @property
    def measured_pfa(self) -> float:
        """Fraction of clutter pixels the test fired on; NaN if there were none."""
        if self.clutter_pixels == 0:
            return float("nan")
        return self.clutter_hits / self.clutter_pixels

    @property
    def excess(self) -> float:
        """How many times the requested rate the clutter actually produced."""
        return self.measured_pfa / self.requested_pfa


def measure_clutter_false_alarms(
    amplitude: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    pfas: Sequence[float] = DEFAULT_SWEEP,
    reference: CfarConfig | None = None,
    exclusion_px: int = DEFAULT_EXCLUSION_PX,
) -> tuple[PfaMeasurement, ...]:
    """Measure the pixel false-alarm rate on clutter for each setting in ``pfas``.

    ``mask`` marks pixels that were never searched: nodata, land, platforms.
    ``reference`` fixes which objects are excluded; rows differ only in Pfa.
    """
    if exclusion_px < 0:
        raise ValueError(f"exclusion_px must be non-negative, got {exclusion_px}")
    reference = reference or CfarConfig()
    unsearched = np.zeros(amplitude.shape, dtype=bool) if mask is None else mask

    objects = _binary_opening(cfar_detect(amplitude, reference, mask=mask))
    clutter = ~unsearched & ~dilate(objects, exclusion_px)
    clutter_pixels = int(clutter.sum())

    measurements = []
    for pfa in pfas:
        config = replace(reference, pfa=pfa)
        hits = cfar_detect(amplitude, config, mask=mask)
        measurements.append(
            PfaMeasurement(
                requested_pfa=pfa,
                alpha=config.alpha,
                clutter_pixels=clutter_pixels,
                clutter_hits=int((hits & clutter).sum()),
                candidates=len(detect_candidates(amplitude, config=config, mask=mask)),
            )
        )
    return tuple(measurements)
