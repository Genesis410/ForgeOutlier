"""Feature extraction backbone architectures."""

from typing import Union
import torch
import torch.nn as nn
import numpy as np
from forgeoutlier.core.exceptions import NumericalInstabilityError
from forgeoutlier.core.config import NUMERICAL_CONSTANTS


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for MobileNetV3."""

    def __init__(self, in_channels: int, squeeze_factor: int = 4) -> None:
        super().__init__()
        squeezed = max(1, in_channels // squeeze_factor)
        self.fc1 = nn.Conv2d(in_channels, squeezed, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezed, in_channels, kernel_size=1)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale


class InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_dim: int,
        use_se: bool,
        use_hs: bool,
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        act_layer = nn.Hardswish if use_hs else nn.ReLU

        layers = []
        # Expansion
        if expansion_dim != in_channels:
            layers.append(nn.Conv2d(in_channels, expansion_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(expansion_dim))
            layers.append(act_layer(inplace=True))

        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.append(
            nn.Conv2d(
                expansion_dim,
                expansion_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=expansion_dim,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(expansion_dim))
        layers.append(act_layer(inplace=True))

        # SE
        if use_se:
            layers.append(SqueezeExcitation(expansion_dim))

        # Pointwise projection
        layers.append(nn.Conv2d(expansion_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3-Small backbone outputting penultimate pre-logits (D=576 or D=1024).

    Strictly offline, pure PyTorch implementation without external torchvision dependency.
    """

    def __init__(self, feature_dim: int = 576) -> None:
        super().__init__()
        self._feature_dim = feature_dim

        # Layer specs for MobileNetV3-Small:
        # [kernel, exp_size, out_channels, SE, HS, stride]
        cfg = [
            [3, 16, 16, True, False, 2],
            [3, 72, 24, False, False, 2],
            [3, 88, 24, False, False, 1],
            [5, 96, 40, True, True, 2],
            [5, 240, 40, True, True, 1],
            [5, 240, 40, True, True, 1],
            [5, 120, 48, True, True, 1],
            [5, 144, 48, True, True, 1],
            [5, 288, 96, True, True, 2],
            [5, 576, 96, True, True, 1],
            [5, 576, 96, True, True, 1],
        ]

        # Initial stem
        layers = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        ]

        in_ch = 16
        for k, exp, out_ch, se, hs, s in cfg:
            layers.append(
                InvertedResidual(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=s,
                    expansion_dim=exp,
                    use_se=se,
                    use_hs=hs,
                )
            )
            in_ch = out_ch

        # Final stem to 576 channels
        layers.append(nn.Conv2d(in_ch, 576, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(576))
        layers.append(nn.Hardswish(inplace=True))

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Pre-logits projection if feature_dim == 1024
        if feature_dim == 1024:
            self.pre_logits = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(inplace=True),
            )
        else:
            self.pre_logits = nn.Identity()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pre-logits embedding (B, D)."""
        feats = self.features(x)
        pooled = self.gap(feats).flatten(1)
        z = self.pre_logits(pooled)
        return z
