"""
streams.py
----------
The two modality encoders. Both consume the tensors produced by
preprocessing/pipeline.py and emit per-frame embedding sequences of
identical shape (B, T, EMBED_DIM) so the fusion layer can compare
them frame-by-frame.

    VideoLipStream : (B, T, 1, 96, 96)  ->  (B, T, 256)
    AudioStream    : (B, 1, 80, 4T)     ->  (B, T, 256)

Design notes
------------
* The lip encoder uses a 3D-conv stem (captures short-range motion —
  the actual articulation dynamics) followed by per-frame 2D ResNet
  blocks. Its final conv layer keeps a spatial map (B*T, C, 6, 6)
  that Phase 3 hooks for Grad-CAM.
* The audio encoder downsamples the mel time axis by exactly 4x
  (stride 2 twice), so its output length equals the video frame
  count T. This is why preprocessing enforced 4 mel cols per frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EMBED_DIM = 256


# ----------------------------------------------------------------------
# Shared building block
# ----------------------------------------------------------------------
class ResBlock2d(nn.Module):
    """Standard pre-activation 2D residual block with optional stride."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


# ----------------------------------------------------------------------
# Video / lip stream
# ----------------------------------------------------------------------
class VideoLipStream(nn.Module):
    """
    (B, T, 1, 96, 96) -> (B, T, EMBED_DIM)

    Exposes `self.last_conv_maps` after each forward pass:
    a (B, T, 256, 6, 6) tensor of spatial features used by Grad-CAM.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        # 3D stem: temporal kernel 5 captures ~200 ms of lip motion.
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),          # -> (B, 32, T, 24, 24)
        )
        # Per-frame 2D trunk (applied with frames folded into batch).
        self.trunk = nn.Sequential(
            ResBlock2d(32, 64),
            ResBlock2d(64, 128, stride=2),    # 24 -> 12
            ResBlock2d(128, 256, stride=2),   # 12 -> 6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, embed_dim)

        self.last_conv_maps: torch.Tensor | None = None  # for Grad-CAM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = x.transpose(1, 2)                 # (B, 1, T, 96, 96)
        x = self.stem(x)                      # (B, 32, T, 24, 24)
        x = x.transpose(1, 2).reshape(B * T, 32, 24, 24)
        maps = self.trunk(x)                  # (B*T, 256, 6, 6)

        # Keep the graph-connected spatial maps for Grad-CAM hooks.
        self.last_conv_maps = maps.view(B, T, 256, 6, 6)

        z = self.pool(maps).flatten(1)        # (B*T, 256)
        z = self.proj(z).view(B, T, -1)       # (B, T, EMBED_DIM)
        return z


# ----------------------------------------------------------------------
# Audio stream
# ----------------------------------------------------------------------
class AudioStream(nn.Module):
    """
    (B, 1, 80, 4T) -> (B, T, EMBED_DIM)

    Time axis is downsampled exactly 4x (two stride-2 convs) so the
    output sequence aligns 1:1 with video frames. The mel-frequency
    axis is fully pooled at the end.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        # (freq_stride, time_stride) chosen so time shrinks by 4 total.
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 1), padding=1, bias=False),   # 80->40
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock2d(32, 64, stride=1),
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1, bias=False),  # 40->20, T/2
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock2d(64, 128, stride=1),
            nn.Conv2d(128, 128, 3, stride=(2, 2), padding=1, bias=False),# 20->10, T/4
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock2d(128, 256, stride=1),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # pool freq, keep time
        self.proj = nn.Linear(256, embed_dim)

        self.last_conv_maps: torch.Tensor | None = None   # (B, 256, F', T) for audio CAM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)                    # (B, 256, 10, T)
        self.last_conv_maps = feats
        z = self.freq_pool(feats).squeeze(2)   # (B, 256, T)
        z = z.transpose(1, 2)                  # (B, T, 256)
        return self.proj(z)


if __name__ == "__main__":
    B, T = 2, 25
    lips = torch.randn(B, T, 1, 96, 96)
    mel = torch.randn(B, 1, 80, 4 * T)

    v, a = VideoLipStream(), AudioStream()
    zv, za = v(lips), a(mel)
    print("video embeddings:", tuple(zv.shape))       # (2, 25, 256)
    print("audio embeddings:", tuple(za.shape))       # (2, 25, 256)
    print("gradcam maps    :", tuple(v.last_conv_maps.shape))  # (2, 25, 256, 6, 6)
