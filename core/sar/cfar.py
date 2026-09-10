"""Cell-Averaging CFAR detection for SAR imagery.

CFAR - Constant False Alarm Rate - is the classical maritime radar detector.
For every pixel it estimates the local sea clutter from a ring of training
cells around it, skipping a guard band so the target's own energy does not
contaminate its background estimate, and declares a detection when the pixel
exceeds a multiple of that local mean.

Why it earns a place in a deep-learning project
-----------------------------------------------
It needs no training data, so it works on day one. Its false-alarm rate is a
tunable parameter rather than an emergent property, which makes it an
interpretable baseline to measure a learned detector against. And a
Sentinel-1 scene is >99.9% empty ocean, so running CFAR first cuts the tiles a
CNN must look at by one to two orders of magnitude - which is what makes
inference feasible on a CPU.

Implementation note: the ring sums are computed with summed-area tables, so
the cost per pixel is constant regardless of window size. A naive sliding
window over a 25,000 x 17,000 scene is not viable.

Statistics: for fully-developed speckle, homogeneous sea clutter imaged with
``L`` looks has Gamma-distributed intensity with shape ``L``. The threshold
multiplier for a desired false-alarm probability is therefore

    alpha = Q_inv(L, Pfa) / L

where ``Q_inv`` inverts the regularised upper incomplete gamma function.

The single-look case (L = 1) reduces to the familiar exponential result
``alpha = -ln(Pfa)``, and that simpler formula is what most CFAR write-ups
quote. Applying it to multi-look data is a real trap: Sentinel-1 IW GRDH is
about 4.4 looks, whose intensity distribution is far less heavy-tailed, so the
exponential formula sets the threshold roughly twice as high as required. The
detector then runs far more conservatively than its Pfa dial claims - missing
vessels while appearing correctly configured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Defaults for Sentinel-1 IW GRDH at 10 m.
#
# The guard band must exceed the target's LENGTH, not its beam. This is the
# easiest CFAR mistake to make and it fails quietly: if a 300 m ship is longer
# than the guard box, its own bright pixels fall in its training ring, inflate
# its local background, and suppress its own detection. The symptom is a large
# vessel detected only as a few fragments around its edges.
#
# 24 px of guard gives a 490 m box, clearing the longest vessel afloat. The
# cost is a slightly less local clutter estimate, which is the right trade:
# integral images make the window size free, so the only price is adaptivity.
DEFAULT_GUARD_HALF = 24  # 24 px -> 490 m guard box side
DEFAULT_TRAIN_HALF = 40  # 40 px -> 810 m outer box side
DEFAULT_PFA = 1e-6

# Equivalent number of looks for Sentinel-1 IW GRDH. Multi-looking averages
# independent speckle samples, which narrows the intensity distribution.
IW_GRDH_LOOKS = 4.4

# Below this the background estimate is too noisy for the threshold to mean
# anything, whatever the arithmetic says.
MIN_TRAINING_CELLS = 32


def _regularised_upper_gamma(a: float, x: float) -> float:
    """Q(a, x): the probability a Gamma(a, 1) variate exceeds x.

    Series expansion below the crossover, continued fraction above, which is
    the standard numerically stable split. Implemented here rather than pulled
    from scipy to keep the detector's hot path free of a heavy dependency.
    """
    if x <= 0.0:
        return 1.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)

    if x < a + 1.0:
        # Series for the LOWER incomplete gamma, then complement.
        term = 1.0 / a
        total = term
        for n in range(1, 1000):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(log_prefactor)

    # Lentz's continued fraction for the upper incomplete gamma.
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(log_prefactor) * h


def gamma_threshold(looks: float, pfa: float) -> float:
    """Solve Q(looks, looks * alpha) = pfa for the multiplier alpha.

    Bisection: the target function is monotonic, the bracket is easy to
    establish, and this runs once per configuration rather than per pixel.
    """
    if looks <= 0.0:
        raise ValueError("looks must be positive")

    def excess(alpha: float) -> float:
        return _regularised_upper_gamma(looks, looks * alpha) - pfa

    low, high = 0.0, 1.0
    while excess(high) > 0.0:
        high *= 2.0
        if high > 1e6:  # pragma: no cover - unreachable for sane Pfa
            raise ValueError(f"no threshold found for pfa={pfa}")
    for _ in range(200):
        mid = 0.5 * (low + high)
        if excess(mid) > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


@dataclass(frozen=True)
class CfarConfig:
    """CFAR parameters.

    ``satellite`` is carried because these do NOT transfer between platforms.
    Global Fishing Watch's production detector required full recalibration for
    S1C/S1D and was offline for two months as a result.
    """

    guard_half: int = DEFAULT_GUARD_HALF
    train_half: int = DEFAULT_TRAIN_HALF
    pfa: float = DEFAULT_PFA
    looks: float = IW_GRDH_LOOKS
    satellite: str | None = None

    def __post_init__(self) -> None:
        if self.guard_half < 0:
            raise ValueError("guard_half must be non-negative")
        if self.train_half <= self.guard_half:
            raise ValueError(
                f"train_half ({self.train_half}) must exceed "
                f"guard_half ({self.guard_half}); otherwise there are no "
                "training cells to estimate clutter from"
            )
        if not 0.0 < self.pfa < 1.0:
            raise ValueError(f"pfa must be in (0, 1), got {self.pfa}")
        if self.looks <= 0.0:
            raise ValueError(f"looks must be positive, got {self.looks}")
        if self.training_cells < MIN_TRAINING_CELLS:
            raise ValueError(f"only {self.training_cells} training cells; widen train_half")

    @property
    def outer_size(self) -> int:
        return 2 * self.train_half + 1

    @property
    def guard_size(self) -> int:
        return 2 * self.guard_half + 1

    @property
    def training_cells(self) -> int:
        """Cells in the ring: the outer box minus the guard box."""
        return self.outer_size**2 - self.guard_size**2

    @property
    def alpha(self) -> float:
        """Threshold multiplier for the configured false-alarm probability.

        Derived from the Gamma clutter model for the configured number of
        looks. With many training cells the extra loss from estimating the
        background is negligible, so it is not modelled here.
        """
        return gamma_threshold(self.looks, self.pfa)


def _integral_image(array: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row and column, for O(1) box sums."""
    padded = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = array.cumsum(axis=0).cumsum(axis=1)
    return padded


def _box_sums(integral: np.ndarray, size: int, rows: int, cols: int) -> np.ndarray:
    """Sum over every ``size`` x ``size`` box, aligned to the output grid."""
    return (
        integral[size : size + rows, size : size + cols]
        - integral[0:rows, size : size + cols]
        - integral[size : size + rows, 0:cols]
        + integral[0:rows, 0:cols]
    )


def clutter_mean(intensity: np.ndarray, config: CfarConfig) -> np.ndarray:
    """Local background mean from the training ring around each pixel.

    The guard band is excluded so a vessel does not inflate its own threshold
    and hide itself - the single most important detail in a CFAR.
    """
    pad = config.train_half
    # Reflect rather than zero-pad: zeros would fabricate a dark background at
    # the scene edge and produce a ring of false detections there.
    padded = np.pad(intensity.astype(np.float64), pad, mode="reflect")
    rows, cols = intensity.shape

    integral = _integral_image(padded)
    outer = _box_sums(integral, config.outer_size, rows, cols)

    # The guard box shares the same centre, so it is offset inwards.
    offset = config.train_half - config.guard_half
    guard_integral = integral[offset:, offset:]
    guard = _box_sums(guard_integral, config.guard_size, rows, cols)

    return (outer - guard) / config.training_cells


def cfar_detect(
    amplitude: np.ndarray,
    config: CfarConfig | None = None,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean map of CFAR detections.

    ``amplitude`` is raw Sentinel-1 GRD digital numbers. These are amplitude,
    so they are squared to intensity here: the exponential clutter model the
    threshold is derived from applies to intensity, not amplitude.

    ``mask`` marks pixels to exclude (land, fixed infrastructure). Masked
    pixels are removed from the result AND from the clutter estimate, because
    bright land inside a training ring would otherwise raise the threshold and
    blind the detector along the entire coastline.
    """
    config = config or CfarConfig()
    intensity = amplitude.astype(np.float64) ** 2

    if mask is not None:
        if mask.shape != amplitude.shape:
            raise ValueError(f"mask shape {mask.shape} does not match image {amplitude.shape}")
        # Replace excluded pixels with the scene's own sea level, so they
        # neither raise nor depress the local background estimate.
        sea_level = float(np.median(intensity[~mask])) if (~mask).any() else 0.0
        intensity = np.where(mask, sea_level, intensity)

    background = clutter_mean(intensity, config)
    detections = intensity > config.alpha * background

    if mask is not None:
        detections &= ~mask
    return detections


def measured_false_alarm_rate(detections: np.ndarray, truth: np.ndarray | None = None) -> float:
    """Fraction of pixels declared detections where no target exists.

    With no truth map every detection counts as a false alarm, which is the
    right measure on a target-free calibration scene.
    """
    if truth is None:
        return float(detections.mean())
    clutter = ~truth
    if not clutter.any():
        return 0.0
    return float((detections & clutter).sum() / clutter.sum())
