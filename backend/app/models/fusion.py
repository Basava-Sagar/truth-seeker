"""
fusion.py
---------
Cross-modal fusion head and the full TruthSeekerNet.

The fusion layer detects desynchronization with two complementary
mechanisms, both of which produce explainable byproducts:

1. Cross-modal attention. Each audio frame attends over ALL video
   frames (and vice versa). In a genuine video the attention matrix
   is strongly diagonal — audio at time t matches lips at time t.
   Lip-sync fakes produce blurred or off-diagonal attention. We keep
   the (T x T) attention maps for Phase 3 visualization.

2. Per-frame sync distance. Cosine distance between the aligned
   lip/audio embeddings at each t. Spikes in this curve localize
   *when* the modalities disagree — the temporal explainability
   signal shown on the frontend timeline.

Output: probability the window is FAKE, plus an `aux` dict of every
intermediate needed by the explainability module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .streams import EMBED_DIM, AudioStream, VideoLipStream


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positions so attention knows frame order."""

    def __init__(self, dim: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                      # (B, T, D)
        return x + self.pe[:, : x.size(1)]


class CrossModalBlock(nn.Module):
    """
    One round of bidirectional cross-attention + feed-forward.
    Returns updated (video, audio) sequences and both attention maps.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.a2v = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.v2a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_v1, self.norm_a1 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm_v2, self.norm_a2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff_v = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.ff_a = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, zv, za):
        # Video queries attend to audio keys/values, and vice versa.
        v_upd, attn_v2a = self.a2v(zv, za, za, need_weights=True,
                                   average_attn_weights=True)
        a_upd, attn_a2v = self.v2a(za, zv, zv, need_weights=True,
                                   average_attn_weights=True)
        zv = self.norm_v1(zv + v_upd)
        za = self.norm_a1(za + a_upd)
        zv = self.norm_v2(zv + self.ff_v(zv))
        za = self.norm_a2(za + self.ff_a(za))
        return zv, za, attn_v2a, attn_a2v      # attn: (B, T, T)


class TruthSeekerNet(nn.Module):
    """
    Full model: dual streams -> cross-modal fusion -> fake probability.

    forward(lips, mel) returns (logits, aux) where aux contains:
        sync_distance : (B, T)    per-frame cosine distance curve
        attn_v2a      : (B, T, T) video-queries-audio attention (last block)
        attn_a2v      : (B, T, T) audio-queries-video attention (last block)
    """

    def __init__(self, dim: int = EMBED_DIM, n_blocks: int = 2):
        super().__init__()
        self.video_stream = VideoLipStream(dim)
        self.audio_stream = AudioStream(dim)
        self.pos = PositionalEncoding(dim)
        self.blocks = nn.ModuleList(CrossModalBlock(dim) for _ in range(n_blocks))

        # Classifier sees pooled fused features + sync statistics.
        self.head = nn.Sequential(
            nn.Linear(dim * 2 + 3, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, lips: torch.Tensor, mel: torch.Tensor):
        zv_raw = self.video_stream(lips)       # (B, T, D)
        za_raw = self.audio_stream(mel)        # (B, T, D)

        # --- Explainability signal #2: per-frame sync distance ---
        # Computed on the raw (pre-fusion) embeddings so it purely
        # reflects modality agreement, not classifier shortcuts.
        sync_dist = 1 - F.cosine_similarity(zv_raw, za_raw, dim=-1)  # (B, T)

        zv, za = self.pos(zv_raw), self.pos(za_raw)
        attn_v2a = attn_a2v = None
        for blk in self.blocks:
            zv, za, attn_v2a, attn_a2v = blk(zv, za)

        # --- Pooled features + sync statistics -> verdict ---
        fused = torch.cat([zv.mean(1), za.mean(1)], dim=-1)          # (B, 2D)
        stats = torch.stack([sync_dist.mean(1),
                             sync_dist.max(1).values,
                             sync_dist.std(1)], dim=-1)              # (B, 3)
        logits = self.head(torch.cat([fused, stats], dim=-1)).squeeze(-1)

        aux = {
            "sync_distance": sync_dist,
            "attn_v2a": attn_v2a,
            "attn_a2v": attn_a2v,
        }
        return logits, aux

    @torch.no_grad()
    def predict_proba(self, lips, mel):
        """Convenience for inference: returns P(fake) in [0, 1]."""
        logits, aux = self.forward(lips, mel)
        return torch.sigmoid(logits), aux


def truth_seeker_loss(logits, labels, aux, sync_weight: float = 0.1):
    """
    Training loss = BCE on the verdict + a weak auxiliary term that
    pushes sync_distance UP on fakes and DOWN on genuine clips. This
    keeps the explainability curve honest: it is directly supervised
    to mean what we tell the user it means.
    labels: (B,) float, 1 = fake, 0 = real.
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    mean_dist = aux["sync_distance"].mean(1)          # (B,)
    # Hinge-style: fakes should exceed margin, reals should stay below.
    sync_term = (labels * F.relu(0.4 - mean_dist)
                 + (1 - labels) * F.relu(mean_dist - 0.2)).mean()
    return bce + sync_weight * sync_term


if __name__ == "__main__":
    B, T = 2, 25
    model = TruthSeekerNet()
    logits, aux = model(torch.randn(B, T, 1, 96, 96), torch.randn(B, 1, 80, 4 * T))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.1f}M")
    print("logits        :", tuple(logits.shape))                 # (2,)
    print("sync_distance :", tuple(aux['sync_distance'].shape))   # (2, 25)
    print("attn_v2a      :", tuple(aux['attn_v2a'].shape))        # (2, 25, 25)
    loss = truth_seeker_loss(logits, torch.tensor([1.0, 0.0]), aux)
    loss.backward()
    print("loss ok       :", float(loss))
