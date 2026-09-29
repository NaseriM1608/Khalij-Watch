"""Stride-2 dense point detector: timm encoder, U-Net decoder, objectness + offset heads.

The xView3-winning design (PLAN.md section 7.1): no boxes, no custom operators,
so ONNX export is a plain conv stack and non-maximum suppression is a 3x3
max-pool done outside the graph. Length, width, heading and class heads are
left out until the dataset carries trustworthy targets for them (PLAN.md
section 17, 2026-09-27: GFW classes changed and CFAR lengths are noisy).
"""

from __future__ import annotations

import math

import timm
import torch
from torch import nn
from torch.nn import functional

# Initial objectness prior: 1% of output cells hold a vessel (CenterNet-style bias).
OBJECTNESS_PRIOR = 0.01


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    """Upsample x2, join the skip connection, two 3x3 convolutions."""

    def __init__(self, cin: int, skip: int, cout: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(_conv_bn_relu(cin + skip, cout), _conv_bn_relu(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convs(torch.cat([x, skip], dim=1))


def _head(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(cin, cout, 1)
    )


class PointDetector(nn.Module):
    """Two-band SAR in, (objectness logits, offsets) at half resolution out."""

    def __init__(
        self,
        encoder: str = "efficientnet_b0",
        *,
        pretrained: bool = False,
        in_chans: int = 2,
        decoder_channels: tuple[int, ...] = (128, 64, 48, 32),
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            encoder, features_only=True, pretrained=pretrained, in_chans=in_chans
        )
        channels = self.encoder.feature_info.channels()
        reductions = self.encoder.feature_info.reduction()
        if reductions[0] != 2 or len(decoder_channels) != len(channels) - 1:
            raise ValueError(
                f"encoder reductions {reductions} do not suit {len(decoder_channels)} decoder blocks"
            )
        blocks = []
        cin = channels[-1]
        for skip, cout in zip(reversed(channels[:-1]), decoder_channels, strict=True):
            blocks.append(DecoderBlock(cin, skip, cout))
            cin = cout
        self.decoder = nn.ModuleList(blocks)
        self.objectness = _head(cin, 1)
        self.offset = _head(cin, 2)
        nn.init.constant_(
            self.objectness[-1].bias, -math.log((1 - OBJECTNESS_PRIOR) / OBJECTNESS_PRIOR)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        y = features[-1]
        for block, skip in zip(self.decoder, reversed(features[:-1]), strict=True):
            y = block(y, skip)
        return self.objectness(y), self.offset(y)
