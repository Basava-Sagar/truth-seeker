"""
eval_dfdc.py -- cross-dataset generalization check.

Runs the already-trained TruthSeekerNet (no training happens here) on a
DFDC sample set to see whether it generalizes beyond FakeAVCeleb's
specific generation artifacts, or whether it overfit to them.

Expects Kaggle's "train_sample_videos.zip" already extracted, e.g.:

    E:\\TruthSeeker-SSD\\DFDC_sample\\train_sample_videos\\
        aagfhgtpmv.mp4
        aapnvogymq.mp4
        ...
        metadata.json

metadata.json maps filename -> {"label": "REAL"|"FAKE", ...}.

Usage (from backend/, with the venv active):
    python eval_dfdc.py "E:\\TruthSeeker-SSD\\DFDC_sample\\train_sample_videos" --limit 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from app.inference import get_model, BATCH
from app.preprocessing.pipeline import preprocess_video


def score_video(model, video_path: Path) -> float | None:
    """Returns P(fake) in [0,1] for one video, or None if preprocessing failed."""
    try:
        pv = preprocess_video(video_path)
    except RuntimeError as e:
        print(f"  [skip] {video_path.name}: {e}")
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
    # Same aggregation as analyze_video(): mean of top-25% most suspicious windows.
    k = max(1, round(len(probs_arr) * 0.25))
    return float(np.sort(probs_arr)[-k:].mean())


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Manual AUC (probability a random positive scores higher than a random
    negative) -- avoids adding sklearn as a new dependency for one script."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-sum formula, equivalent to the pairwise comparison but O(n log n).
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Folder containing DFDC sample mp4s + metadata.json")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of videos evaluated (default: all)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"metadata.json not found in {data_dir}")

    metadata = json.loads(meta_path.read_text())
    items = list(metadata.items())
    if args.limit:
        items = items[: args.limit]

    model, trained = get_model()
    if not trained:
        sys.exit("No trained checkpoint found (app/models/weights/truth_seeker.pt) "
                  "-- refusing to evaluate an untrained model.")

    labels, scores, names = [], [], []
    t0 = time.time()
    for i, (fname, info) in enumerate(items):
        video_path = data_dir / fname
        if not video_path.exists():
            continue
        label = 1 if info["label"].upper() == "FAKE" else 0
        prob = score_video(model, video_path)
        if prob is None:
            continue
        labels.append(label)
        scores.append(prob)
        names.append(fname)
        verdict = "fake" if prob >= args.threshold else "real"
        correct = "OK" if (prob >= args.threshold) == bool(label) else "WRONG"
        print(f"[{i+1}/{len(items)}] {fname}: true={'FAKE' if label else 'REAL':4s} "
              f"pred={verdict:4s} p={prob:.3f}  {correct}")

    if not scores:
        sys.exit("No videos were successfully scored.")

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    preds = (scores_arr >= args.threshold).astype(int)
    accuracy = float((preds == labels_arr).mean())
    auc = roc_auc(labels_arr, scores_arr)

    n_real = int((labels_arr == 0).sum())
    n_fake = int((labels_arr == 1).sum())
    elapsed = time.time() - t0

    print()
    print("=" * 50)
    print(f"DFDC cross-dataset eval  ({len(scores)} videos: {n_real} real, "
          f"{n_fake} fake; {elapsed:.0f}s)")
    print(f"  accuracy : {accuracy:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print("=" * 50)

    out = {
        "n_videos": len(scores), "n_real": n_real, "n_fake": n_fake,
        "accuracy": accuracy, "auc": auc,
        "results": [{"file": n, "label": l, "prob": p}
                     for n, l, p in zip(names, labels, scores)],
    }
    with open("dfdc_eval_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("full results written to dfdc_eval_result.json")


if __name__ == "__main__":
    main()
