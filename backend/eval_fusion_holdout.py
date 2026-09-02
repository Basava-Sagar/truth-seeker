"""
eval_fusion_holdout.py -- does fusion actually beat lip-sync alone?
------------------------------------------------------------------------
Runs THREE things against the exact same DFDC permanent holdout set
(holdout_manifest.json, written by train_visual_combined.py -- never
touched by any training, visual or lip-sync):

    1. lip-sync-only (the original, unmodified TruthSeekerNet)
    2. visual-only (the combined-trained FaceManipulationNet)
    3. several fusion strategies (weighted average at a few weights, plus
       the max rule) via inference_fusion.py

This is the actual decision point: if no fusion strategy beats #1 by a
meaningful margin, the honest conclusion is "use lip-sync alone" -- fusion
adds complexity (two models to maintain, ship, explain) that isn't
justified unless it demonstrably helps. Don't assume; check.

Usage (from backend/, after train_visual_combined.py has completed):
    python eval_fusion_holdout.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from app.inference import get_model as get_lipsync_model
from app.inference_fusion import score_visual, fuse_scores
from app.preprocessing.pipeline import preprocess_video

HOLDOUT_MANIFEST = Path("holdout_manifest.json")
BATCH = 8


def score_lipsync(video_path: Path) -> float | None:
    """Lightweight lip-sync-only scoring -- same top-25%-window aggregation
    as analyze_video()/eval_dfdc.py's score_video, but skips the full
    Grad-CAM explanation machinery for speed across many videos. Produces
    the same probability analyze_video() would report, just faster to
    compute in bulk."""
    model, trained = get_lipsync_model()
    if not trained:
        return None
    try:
        pv = preprocess_video(video_path)
    except RuntimeError:
        return None

    probs = []
    with torch.no_grad():
        for i in range(0, len(pv.windows), BATCH):
            chunk = pv.windows[i:i + BATCH]
            lips = torch.from_numpy(np.stack([w.lips for w in chunk]))
            mel = torch.from_numpy(np.stack([w.mel for w in chunk]))
            p, _aux = model.predict_proba(lips, mel)
            probs.extend(p.tolist())

    probs_arr = np.array(probs)
    k = max(1, round(len(probs_arr) * 0.25))
    return float(np.sort(probs_arr)[-k:].mean())


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    if not HOLDOUT_MANIFEST.exists():
        sys.exit(f"{HOLDOUT_MANIFEST} not found -- run train_visual_combined.py first.")

    manifest = json.loads(HOLDOUT_MANIFEST.read_text())
    files, labels = manifest["files"], manifest["labels"]
    print(f"holdout set: {len(files)} videos "
          f"({labels.count(0)} real / {labels.count(1)} fake)\n")

    rows = []
    for i, (fpath, label) in enumerate(zip(files, labels), 1):
        video_path = Path(fpath)
        p_lip = score_lipsync(video_path)
        p_vis = score_visual(video_path)
        if p_lip is None:
            print(f"  [{i}/{len(files)}] skip {video_path.name}: lip-sync preprocessing failed")
            continue
        rows.append((label, p_lip, p_vis))
        if i % 25 == 0:
            print(f"  ...{i}/{len(files)} scored")

    if not rows:
        sys.exit("No holdout videos were successfully scored by the lip-sync model.")

    labels_arr = np.array([r[0] for r in rows])
    p_lip_arr = np.array([r[1] for r in rows])

    print(f"\nscored {len(rows)}/{len(files)} holdout videos\n")

    print("=" * 70)
    print(f"{'strategy':30s} {'AUC':>10s}")
    print("-" * 70)

    auc_lip = roc_auc(labels_arr, p_lip_arr)
    print(f"{'lip-sync only (baseline)':30s} {auc_lip:>10.4f}")

    p_vis_arr = np.array([r[2] if r[2] is not None else np.nan for r in rows])
    vis_mask = ~np.isnan(p_vis_arr)
    if vis_mask.sum() > 0:
        auc_vis = roc_auc(labels_arr[vis_mask], p_vis_arr[vis_mask])
        print(f"{'visual only (combined-trained)':30s} {auc_vis:>10.4f}  "
              f"({vis_mask.sum()}/{len(rows)} had a usable face crop)")
    else:
        print(f"{'visual only':30s} {'n/a':>10s} (no usable face crops)")

    best = (auc_lip, "lip-sync only (baseline)")
    for method, weight in [("weighted", 0.3), ("weighted", 0.5),
                           ("weighted", 0.7), ("max", None)]:
        fused = np.array([
            fuse_scores(p_lip, p_vis, method=method, weight=weight or 0.5)
            for p_lip, p_vis in zip(p_lip_arr, [r[2] for r in rows])
        ])
        auc_f = roc_auc(labels_arr, fused)
        label = f"fused ({method}" + (f", w={weight})" if weight else ")")
        print(f"{label:30s} {auc_f:>10.4f}")
        if auc_f > best[0]:
            best = (auc_f, label)

    print("=" * 70)
    print(f"\nBEST: {best[1]}  (AUC {best[0]:.4f})")
    if best[1] == "lip-sync only (baseline)":
        print("\n-> No fusion strategy beat lip-sync alone on this holdout set.")
        print("  Honest recommendation: SHIP LIP-SYNC ONLY. Fusion adds real")
        print("  complexity (two models, two checkpoints, harder-to-explain")
        print("  verdicts) that isn't earning its keep here -- document the")
        print("  visual-detector work as a documented experiment/limitation,")
        print("  not a deployed feature.")
    else:
        margin = best[0] - auc_lip
        print(f"\n-> Beats lip-sync-only baseline by {margin:+.4f} AUC.")
        print("  Worth deploying IF this margin holds up as meaningful given "
              f"the holdout set size ({len(rows)} videos) -- a small holdout "
              "means real sampling noise, so treat a thin margin cautiously.")


if __name__ == "__main__":
    main()
