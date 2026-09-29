"""Losses for the point detector, all respecting the per-pixel ignore weight.

Objectness defaults to the penalty-reduced focal loss the xView3 winner used
for the heatmap. Weighted BCE is kept as the alternative the plan asks to
compare under label noise (PLAN.md section 7.1b). Offsets use L1 on the
peak cells only, as in CenterNet.
"""

from __future__ import annotations

import torch
from torch.nn import functional

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0
_EPS = 1e-6


def reduced_focal_loss(
    logits: torch.Tensor, heatmap: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """CenterNet focal loss over weighted pixels, normalised by the number of peaks."""
    prob = torch.sigmoid(logits).clamp(_EPS, 1 - _EPS)
    peak = (heatmap >= 1 - _EPS).float()
    positive = -((1 - prob) ** FOCAL_ALPHA) * torch.log(prob) * peak
    negative = -((1 - heatmap) ** FOCAL_BETA) * prob**FOCAL_ALPHA * torch.log(1 - prob) * (1 - peak)
    total = ((positive + negative) * weight).sum()
    return total / (peak * weight).sum().clamp(min=1.0)


def weighted_bce_loss(
    logits: torch.Tensor, heatmap: torch.Tensor, weight: torch.Tensor, *, pos_weight: float = 10.0
) -> torch.Tensor:
    """BCE against the Gaussian heatmap over weighted pixels, peaks up-weighted."""
    loss = functional.binary_cross_entropy_with_logits(logits, heatmap, reduction="none")
    boost = 1.0 + (pos_weight - 1.0) * (heatmap >= 1 - _EPS).float()
    return (loss * weight * boost).sum() / weight.sum().clamp(min=1.0)


def offset_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean L1 offset error over peak cells."""
    return (functional.l1_loss(pred, target, reduction="none") * mask).sum() / (
        2 * mask.sum()
    ).clamp(min=1.0)


OBJECTNESS_LOSSES = {"reduced_focal": reduced_focal_loss, "bce": weighted_bce_loss}
