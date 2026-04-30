"""Aria audio CNN — small enough to run in browser via ONNX, big enough
to learn respiratory-distress acoustic patterns.

Architecture:
    input  : (B, 1, MEL_BINS=64, T=63)
    conv1  : 1   -> 16  kernel 3, stride 1, BN, ReLU, MaxPool 2x2
    conv2  : 16  -> 32  kernel 3, stride 1, BN, ReLU, MaxPool 2x2
    conv3  : 32  -> 48  kernel 3, stride 1, BN, ReLU, MaxPool 2x2
    conv4  : 48  -> 64  kernel 3, stride 1, BN, ReLU, AdaptiveAvgPool 1x1
    fc     : 64  -> 32  ReLU
    head   : 32  -> 2   (logits: [normal, distress])

~120k params total, ~470 KB ONNX file.

The penultimate layer (`fc` output) is also exposed as `embed()` so the
in-browser fine-tune ("train on your family") can fit a 1-NN / Mahalanobis
threshold per user without retraining the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aria.audio import MEL_BINS


class AriaModel(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(32, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.embed_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def example_input() -> torch.Tensor:
    """Canonical tensor shape used for ONNX export."""
    return torch.zeros(1, 1, MEL_BINS, 63, dtype=torch.float32)
