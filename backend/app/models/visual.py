"""
visual.py — face-manipulation (blending/artifact) classifier
--------------------------------------------------------------
Phase 3 of the visual-manipulation detector plan. A SEPARATE model from
TruthSeekerNet (fusion.py) -- this one looks only at single face crops
(224x224x3 color, see video_utils.py's FACE_SIZE/face_crop) and asks
"does this face look visually manipulated?", independent of any audio-sync
signal. Intended to be combined with the existing lip-sync model's verdict
at inference time (Phase 4: fusion logic), not to replace it.

Architecture deliberately reuses ResBlock2d from streams.py for consistency
with the rest of the codebase (same block already trained reliably on this
hardware) rather than importing an unfamiliar backbone. Scaled up for the
larger 224x224 color input: an extra downsampling stage and a strided-conv
stem (going straight to full-resolution ResBlocks at 224x224 would be
needlessly expensive) versus the lip stream's smaller 96x96 input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.models.streams import ResBlock2d


class FaceManipulationNet(nn.Module):
    """
    (B, 3, 224, 224) -> logit (B,)  [raw score, use BCEWithLogitsLoss /
    torch.sigmoid for probability -- same convention as TruthSeekerNet]

    Exposes self.last_conv_maps, a (B, 512, 7, 7) spatial feature map, for
    a future Grad-CAM pass on this detector (not wired up yet -- Phase 3
    only needs the classifier itself; explainability for this detector is
    a natural follow-on once it's trained and validated).
    """

    def __init__(self):
        super().__init__()
        # Strided stem: 224 -> 112 (conv) -> 56 (maxpool). Matches the
        # spirit of a standard ResNet stem -- cheap early downsampling
        # before the expensive per-block convolutions kick in.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),   # -> (B, 32, 56, 56)
        )
        self.trunk = nn.Sequential(
            ResBlock2d(32, 64),
            ResBlock2d(64, 128, stride=2),    # 56 -> 28
            ResBlock2d(128, 256, stride=2),   # 28 -> 14
            ResBlock2d(256, 512, stride=2),   # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 1)

        self.last_conv_maps: torch.Tensor | None = None  # (B, 512, 7, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                # (B, 32, 56, 56)
        maps = self.trunk(x)            # (B, 512, 7, 7)
        self.last_conv_maps = maps

        z = self.pool(maps).flatten(1)  # (B, 512)
        logit = self.head(z).squeeze(-1)  # (B,)
        return logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience for inference: returns P(visually manipulated) in [0, 1]."""
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__":
    B = 4
    x = torch.randn(B, 3, 224, 224)
    model = FaceManipulationNet()
    logit = model(x)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    print("logit shape:", tuple(logit.shape))                       # (4,)
    print("gradcam maps:", tuple(model.last_conv_maps.shape))       # (4, 512, 7, 7)
    loss = F.binary_cross_entropy_with_logits(logit, torch.randint(0, 2, (B,)).float())
    print("loss ok:", loss.item())
