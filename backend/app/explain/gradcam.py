"""
gradcam.py  (Phase 3 — Explainability)
--------------------------------------
Turns the model's internals into human-readable evidence:

1. Video Grad-CAM   — which pixels of the mouth drove the verdict,
                      projected back onto the ORIGINAL video frames
                      using the bounding boxes saved in Phase 1.
2. Audio Grad-CAM   — which time-frequency regions of the mel
                      spectrogram drove the verdict, rendered as an
                      overlay on the spectrogram itself.
3. Attention map    — the (T x T) cross-modal attention matrix.
                      Genuine speech -> strong diagonal. Lip-sync
                      fakes -> blurred / shifted mass.
4. Sync curve       — per-frame audio-visual disagreement over time.

Everything is returned as base64 PNG/JPEG strings + plain arrays so
the FastAPI layer (Phase 4) can drop it straight into JSON.

Implementation note: Grad-CAM uses forward/backward hooks on the
last conv trunk of each stream rather than the cached
`last_conv_maps` tensor — hooks are guaranteed to receive gradients
because they sit on the actual computation path.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..models.fusion import TruthSeekerNet
from ..preprocessing.pipeline import AlignedWindow


# ----------------------------------------------------------------------
# Generic Grad-CAM hook pair
# ----------------------------------------------------------------------
class _CamHooks:
    """Captures activations on forward and gradients on backward."""

    def __init__(self, module: torch.nn.Module):
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._h_fwd = module.register_forward_hook(self._on_fwd)
        self._h_bwd = module.register_full_backward_hook(self._on_bwd)

    def _on_fwd(self, _m, _inp, out):
        self.acts = out.detach()

    def _on_bwd(self, _m, _gin, gout):
        self.grads = gout[0].detach()

    def cam(self) -> torch.Tensor:
        """Channel-weighted, ReLU'd class activation map (N, H, W)."""
        weights = self.grads.mean(dim=(2, 3), keepdim=True)     # (N, C, 1, 1)
        return F.relu((weights * self.acts).sum(dim=1))         # (N, H, W)

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------
def _b64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, 82] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise RuntimeError("Image encoding failed")
    return base64.b64encode(buf).decode("ascii")


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def _grab_frames(video_path: str | Path, timestamps: list[float]) -> list:
    """Re-reads specific original frames for heatmap overlays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    cap.release()
    return frames


def _overlay_on_frame(frame: np.ndarray, bbox: tuple, cam: np.ndarray,
                      out_width: int = 480) -> np.ndarray:
    """
    Blends a low-res CAM into the lip bbox region of the original
    frame, draws the box, and downsizes for transport.
    """
    frame = frame.copy()
    x1, y1, x2, y2 = bbox
    cam_img = cv2.resize((_normalize(cam) * 255).astype(np.uint8),
                         (max(x2 - x1, 2), max(y2 - y1, 2)),
                         interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.55, heat, 0.45, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 200, 255), 2)

    h, w = frame.shape[:2]
    scale = out_width / w
    return cv2.resize(frame, (out_width, int(h * scale)))


def _render_attention(attn: np.ndarray, size: int = 300) -> np.ndarray:
    """(T, T) attention matrix -> colormapped image."""
    img = (_normalize(attn) * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)


def _render_mel_cam(mel: np.ndarray, cam: np.ndarray,
                    width: int = 480, height: int = 240) -> np.ndarray:
    """
    mel : (80, T_mel) normalized log-mel
    cam : (F', T')    audio Grad-CAM
    Renders spectrogram (low freq at bottom) with CAM blended in.
    """
    spec = (_normalize(mel) * 255).astype(np.uint8)
    spec = cv2.cvtColor(spec, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(
        (_normalize(cam) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (spec.shape[1], spec.shape[0]),
                      interpolation=cv2.INTER_CUBIC)
    out = cv2.addWeighted(spec, 0.55, heat, 0.45, 0)
    out = cv2.flip(out, 0)  # low frequencies at the bottom
    return cv2.resize(out, (width, height), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
N_HEATMAP_FRAMES = 5


def explain_window(model: TruthSeekerNet,
                   window: AlignedWindow,
                   video_path: str | Path) -> dict:
    """
    Runs one forward+backward pass on a single suspicious window and
    packages every explainability artifact into a JSON-safe dict.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    lips = torch.from_numpy(window.lips).unsqueeze(0)   # (1, T, 1, 96, 96)
    mel = torch.from_numpy(window.mel).unsqueeze(0)     # (1, 1, 80, 100)
    T = lips.shape[1]

    hooks_v = _CamHooks(model.video_stream.trunk[1])   # 12x12 — 4x sharper than
    hooks_a = _CamHooks(model.audio_stream.net)         # the final 6x6 trunk output,
                                                         # still 2 ResBlocks deep so it
                                                         # reflects learned features, not
                                                         # raw stem-level edges/texture.
    try:
        logits, aux = model(lips, mel)
        prob_fake = torch.sigmoid(logits)
        # Backprop from the RAW LOGITS, not the sigmoid probability. When
        # the model is confident (prob_fake near 0 or 1), sigmoid's slope
        # is numerically ~0 at that point, so backpropagating through it
        # kills the gradient before it ever reaches the trunk -- Grad-CAM
        # then computes on all-zero gradients and silently renders a flat,
        # meaningless heatmap instead of failing loudly. The logit itself
        # is unbounded, so its gradient doesn't collapse the same way;
        # this is standard Grad-CAM practice (backprop from the raw class
        # score, before the final activation) for exactly this reason.
        logits.sum().backward()

        cam_video = hooks_v.cam().view(T, 12, 12).cpu().numpy()  # per frame
        cam_audio = hooks_a.cam()[0].cpu().numpy()              # (F', T')
    finally:
        hooks_v.remove()
        hooks_a.remove()

    sync = aux["sync_distance"][0].detach().cpu().numpy()       # (T,)
    attn = aux["attn_a2v"][0].detach().cpu().numpy()            # (T, T)

    # --- Pick the N most influential frames (highest CAM energy) ---
    frame_energy = cam_video.reshape(T, -1).sum(1)
    top_idx = np.sort(np.argsort(frame_energy)[-N_HEATMAP_FRAMES:])
    dt = (window.end_time - window.start_time) / max(T - 1, 1)
    times = [window.start_time + int(i) * dt for i in top_idx]

    originals = _grab_frames(video_path, times)
    heatmap_frames = []
    for i, t, frame in zip(top_idx, times, originals):
        bbox = window.lip_bboxes[int(i)]
        if frame is None or bbox is None:
            continue
        overlay = _overlay_on_frame(frame, bbox, cam_video[int(i)])
        heatmap_frames.append({
            "time": round(float(t), 2),
            "image_b64": _b64(overlay),
        })

    # --- Attention diagonality: fraction of mass within +/-2 frames
    #     of the diagonal. Genuine sync -> high; drift/fakes -> low.
    band = np.abs(np.subtract.outer(np.arange(T), np.arange(T))) <= 2
    diagonality = float((attn * band).sum() / (attn.sum() + 1e-8))

    return {
        "window": {
            "start": round(window.start_time, 2),
            "end": round(window.end_time, 2),
            "fake_probability": round(float(prob_fake.item()), 4),
        },
        "heatmap_frames": heatmap_frames,
        "attention_map_b64": _b64(_render_attention(attn), ".png"),
        "attention_diagonality": round(diagonality, 3),
        "audio_cam_b64": _b64(_render_mel_cam(window.mel[0], cam_audio), ".png"),
        "sync_distance": [round(float(s), 4) for s in sync],
        "frame_importance": [round(float(e), 4)
                             for e in _normalize(frame_energy)],
    }


def build_summary(prob: float, diagonality: float, peak_sync_time: float) -> str:
    """Plain-language explanation string for the frontend."""
    if prob >= 0.5:
        return (
            f"Flagged as likely manipulated ({prob:.0%} confidence). The strongest "
            f"evidence is audio-visual desynchronization peaking at {peak_sync_time:.1f}s — "
            f"mouth movements do not match the speech sounds at that moment. "
            f"Cross-modal attention diagonality is {diagonality:.0%}; genuine speech "
            f"typically stays high. Highlighted regions show which mouth areas and "
            f"audio frequencies drove the decision."
        )
    return (
        f"No manipulation detected ({1 - prob:.0%} confidence in authenticity). "
        f"Lip movements and speech remained consistent throughout, with attention "
        f"diagonality at {diagonality:.0%}. Heatmaps are shown for transparency — "
        f"they indicate what the model examined, not evidence of tampering."
    )
