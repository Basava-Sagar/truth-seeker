"""
inference.py  (Phase 4 — glue layer)
------------------------------------
Single function `analyze_video(path)` that the FastAPI endpoint calls:

    preprocess (Phase 1) -> score every window (Phase 2)
    -> explain the most suspicious window (Phase 3) -> JSON dict.

Model weights are loaded once and cached. If no trained checkpoint is
present, the model runs with random weights and the response is
flagged `model_trained: false` so the frontend can show a warning
instead of presenting noise as a verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .explain.gradcam import build_summary, explain_window
from .models.fusion import TruthSeekerNet
from .preprocessing.pipeline import preprocess_video
from .preprocessing.video_utils import VIDEO_FPS

WEIGHTS_PATH = Path(__file__).parent / "models" / "weights" / "truth_seeker.pt"
FAKE_THRESHOLD = 0.5
BATCH = 8

_model: TruthSeekerNet | None = None
_trained = False


def get_model() -> tuple[TruthSeekerNet, bool]:
    global _model, _trained
    if _model is None:
        _model = TruthSeekerNet()
        if WEIGHTS_PATH.exists():
            state = torch.load(WEIGHTS_PATH, map_location="cpu")
            _model.load_state_dict(state)
            _trained = True
        _model.eval()
    return _model, _trained


def analyze_video(video_path: str | Path) -> dict:
    model, trained = get_model()
    pv = preprocess_video(video_path)          # raises RuntimeError -> 422

    # --- Score all windows in batches (no grad needed here) ---
    probs: list[float] = []
    syncs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(pv.windows), BATCH):
            chunk = pv.windows[i:i + BATCH]
            lips = torch.from_numpy(np.stack([w.lips for w in chunk]))
            mel = torch.from_numpy(np.stack([w.mel for w in chunk]))
            p, aux = model.predict_proba(lips, mel)
            probs.extend(p.tolist())
            syncs.extend(aux["sync_distance"].numpy())

    probs_arr = np.array(probs)

    # Video-level score: mean of the top-25% most suspicious windows.
    # A deepfake segment in a long real video should not be averaged away.
    k = max(1, round(len(probs_arr) * 0.25))
    video_prob = float(np.sort(probs_arr)[-k:].mean())
    verdict = "fake" if video_prob >= FAKE_THRESHOLD else "real"

    # --- Full-video sync timeline (average over overlapping windows) ---
    n_frames = int(round(pv.duration * VIDEO_FPS))
    acc = np.zeros(n_frames)
    cnt = np.zeros(n_frames)
    for w, s in zip(pv.windows, syncs):
        start = int(round(w.start_time * VIDEO_FPS))
        end = min(start + len(s), n_frames)
        acc[start:end] += s[: end - start]
        cnt[start:end] += 1
    mask = cnt > 0
    timeline = [
        {"time": round(i / VIDEO_FPS, 2), "value": round(float(acc[i] / cnt[i]), 4)}
        for i in range(n_frames) if mask[i]
    ]
    peak_time = max(timeline, key=lambda p: p["value"])["time"] if timeline else 0.0

    # --- Deep-dive explanation: pick the window that actually contains
    # peak_time (the moment of worst audio-visual desync), not just the
    # window with the single highest overall confidence. These can differ
    # -- a window's fake_probability reflects the whole fused decision
    # (visual artifacts + sync + attention combined), so the single most
    # confident window doesn't always contain the single worst-desync
    # frame. Explaining the peak-desync window instead keeps the heatmap
    # visually consistent with the "strongest evidence" claim in the
    # summary text. Among windows that overlap peak_time, prefer the most
    # confident one; if none overlap (shouldn't normally happen, since
    # peak_time is derived from these same windows), fall back to the
    # overall most confident window.
    overlapping = [i for i, w in enumerate(pv.windows)
                   if w.start_time <= peak_time <= w.end_time]
    if overlapping:
        top_idx = max(overlapping, key=lambda i: probs_arr[i])
    else:
        top_idx = int(probs_arr.argmax())
    explanation = explain_window(model, pv.windows[top_idx], video_path)
    explanation["summary"] = build_summary(
        video_prob, explanation["attention_diagonality"], peak_time)

    return {
        "verdict": verdict,
        "fake_probability": round(video_prob, 4),
        "model_trained": trained,
        "video": {
            "duration": round(pv.duration, 2),
            "face_detection_rate": round(pv.face_detection_rate, 3),
            "n_windows": len(pv.windows),
        },
        "windows": [
            {
                "start": round(w.start_time, 2),
                "end": round(w.end_time, 2),
                "fake_probability": round(float(p), 4),
            }
            for w, p in zip(pv.windows, probs_arr)
        ],
        "sync_timeline": timeline,
        "explanation": explanation,
    }
