"""
inference_fusion.py — Phase 4: combine lip-sync + visual detectors
----------------------------------------------------------------------
CRITICAL DESIGN GUARANTEE: this module is purely ADDITIVE. It imports
analyze_video() and get_model() from .inference UNCHANGED and calls them
exactly as the existing production code path already does -- nothing here
modifies inference.py, app/models/fusion.py (TruthSeekerNet's
architecture -- note the name collision with this file's *purpose*, not
its content: that module is the lip-sync model's network, this module is
detector-combination logic), or truth_seeker.pt in any way.

Falling back to lip-sync-only is not a special mode -- it's just calling
analyze_video() directly, exactly as the app already does today. Nothing
needs to be "undone" to fall back; the original path was never touched.

Uses the COMBINED-trained visual checkpoint (face_manipulation_combined.pt
-- FakeAVCeleb + real DFDC training data, holdout AUC 0.6907), not the
FakeAVCeleb-only one (face_manipulation.pt, ~random on DFDC) -- see
truth-seeker project notes for why.

Fusion method is deliberately configurable (weighted average, with the
weight left as a parameter) rather than hardcoded, because the right
combination hasn't been empirically validated yet -- see
eval_fusion_holdout.py, which must be run BEFORE wiring this into
production, to confirm fusion actually beats lip-sync-alone's DFDC AUC of
0.6968 rather than assuming it does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .inference import analyze_video, get_model  # noqa: F401 -- re-exported,
                                                   # unchanged, for fallback use
from .models.visual import FaceManipulationNet
from .explain.gradcam_visual import explain_face
from .preprocessing.video_utils import extract_lip_sequence

VISUAL_WEIGHTS_PATH = (Path(__file__).parent / "models" / "weights"
                       / "face_manipulation_combined.pt")
VISUAL_N_FRAMES = 8  # matches dataset_visual.py / eval_dfdc_holdout.py sampling

_visual_model: FaceManipulationNet | None = None
_visual_trained = False


def get_visual_model() -> tuple[FaceManipulationNet, bool]:
    """Loads the COMBINED-trained visual checkpoint. Cached the same way
    inference.py's get_model() caches the lip-sync model -- separate global,
    separate weights file, no shared state between the two."""
    global _visual_model, _visual_trained
    if _visual_model is None:
        _visual_model = FaceManipulationNet()
        if VISUAL_WEIGHTS_PATH.exists():
            state = torch.load(VISUAL_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            _visual_model.load_state_dict(state)
            _visual_trained = True
        _visual_model.eval()
    return _visual_model, _visual_trained


def score_visual(video_path: str | Path, n_frames: int = VISUAL_N_FRAMES) -> float | None:
    """
    P(visually manipulated) for one video, or None if no usable face crops
    were found. Same sampling/aggregation as eval_dfdc_holdout.py's
    score_video, kept in sync deliberately -- so a number computed here
    means the same thing as the 0.6907 holdout result already reported.

    Fast path with no heatmap generation -- used by the eval scripts, which
    iterate over many videos and don't need an image per call. For the
    production single-video path with a heatmap, see explain_visual()
    below, which scores AND explains in one pass rather than duplicating
    the face-extraction work this function also does.
    """
    model, trained = get_visual_model()
    if not trained:
        return None

    try:
        seq = extract_lip_sequence(video_path)
    except Exception:                                  # noqa: BLE001
        return None

    valid = [f for f in seq if f.face_crop is not None]
    if len(valid) < 2:
        return None

    idx = np.linspace(0, len(valid) - 1, num=min(n_frames, len(valid)))
    idx = sorted(set(np.round(idx).astype(int).tolist()))
    faces = np.stack([valid[i].face_crop for i in idx]).astype(np.float32)

    with torch.no_grad():
        x = torch.from_numpy(faces).permute(0, 3, 1, 2)
        p = model.predict_proba(x)
    return float(p.mean().item())


def explain_visual(video_path: str | Path, n_frames: int = VISUAL_N_FRAMES) -> dict | None:
    """
    Scores AND explains a video's visual-manipulation likelihood in one
    pass: samples the same n_frames as score_visual(), finds the single
    MOST suspicious sampled frame (highest individual probability -- same
    "pick the most telling evidence" principle as the lip-sync model's
    peak-desync window selection in inference.py), and runs Grad-CAM on
    just that frame. Returns None under the same conditions score_visual()
    would (no checkpoint, no usable face crops).

    Deliberately a separate function from score_visual() rather than
    adding an optional flag to it -- keeps the fast eval-script path
    (score_visual, used by eval_dfdc_holdout.py / eval_fusion_holdout.py
    across many videos) simple and unchanged, while this heavier path is
    only used for the single-video production endpoint.
    """
    model, trained = get_visual_model()
    if not trained:
        return None

    try:
        seq = extract_lip_sequence(video_path)
    except Exception:                                  # noqa: BLE001
        return None

    valid = [f for f in seq if f.face_crop is not None]
    if len(valid) < 2:
        return None

    idx = np.linspace(0, len(valid) - 1, num=min(n_frames, len(valid)))
    idx = sorted(set(np.round(idx).astype(int).tolist()))
    sampled = [valid[i] for i in idx]
    faces = np.stack([f.face_crop for f in sampled]).astype(np.float32)

    with torch.no_grad():
        x = torch.from_numpy(faces).permute(0, 3, 1, 2)
        per_frame_p = model.predict_proba(x).numpy()

    overall_prob = float(per_frame_p.mean())
    top_i = int(per_frame_p.argmax())

    exp = explain_face(model, sampled[top_i].face_crop)
    return {
        "fake_probability": overall_prob,
        "heatmap_b64": exp["heatmap_b64"],
        "frame_time": sampled[top_i].timestamp,
    }


def fuse_scores(p_lipsync: float, p_visual: float | None,
                method: str = "weighted", weight: float = 0.5) -> float:
    """
    Combines the two detectors' probabilities into one. If the visual
    model produced no usable score (no checkpoint, or no face crops found
    for this specific video), falls back to lip-sync-only automatically --
    this is the SAME fallback principle as the module-level guarantee
    above, just applied per-video instead of as a deployment choice.

    method="weighted": weight*p_visual + (1-weight)*p_lipsync.
    method="max": max(p_lipsync, p_visual) -- flags fake if EITHER detector
        is confident, at the cost of a higher false-positive rate. Worth
        comparing against "weighted" in eval_fusion_holdout.py rather than
        assuming one is better.
    """
    if p_visual is None:
        return p_lipsync

    if method == "max":
        return max(p_lipsync, p_visual)
    elif method == "weighted":
        return weight * p_visual + (1 - weight) * p_lipsync
    else:
        raise ValueError(f"unknown fusion method: {method!r}")


def analyze_video_fused(video_path: str | Path, method: str = "weighted",
                        weight: float = 0.5) -> dict:
    """
    Full production-facing result: calls analyze_video() UNCHANGED to get
    the complete lip-sync explanation (heatmaps, attention map, summary --
    everything the frontend already renders), then adds the visual score
    and a fused verdict alongside it. The original lip-sync result is
    nested under "lipsync" verbatim -- nothing about it is altered,
    filtered, or recomputed. A caller that only wants the original
    behavior can just call analyze_video() directly instead of this
    function; both remain available side by side.
    """
    lipsync_result = analyze_video(video_path)  # untouched existing path
    p_lipsync = lipsync_result["fake_probability"]

    visual = explain_visual(video_path)
    p_visual = visual["fake_probability"] if visual else None
    p_fused = fuse_scores(p_lipsync, p_visual, method=method, weight=weight)

    return {
        "lipsync": lipsync_result,          # exactly what analyze_video() returns, unmodified
        "visual_probability": p_visual,     # None if visual model unavailable/no face found
        "visual_heatmap_b64": visual["heatmap_b64"] if visual else None,
        "visual_heatmap_time": visual["frame_time"] if visual else None,
        "fused_probability": p_fused,
        "fused_verdict": "fake" if p_fused >= 0.5 else "real",
        "fusion_method": method,
        "fusion_weight": weight if method == "weighted" else None,
    }
