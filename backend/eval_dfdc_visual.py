"""
eval_dfdc_visual.py -- cross-dataset generalization check for
FaceManipulationNet specifically (the new visual-only detector, separate
from eval_dfdc.py which tests the original lip-sync TruthSeekerNet).

Purpose: get an early read on whether this detector generalizes beyond
FakeAVCeleb's specific manipulation methods, WITHOUT waiting for all 15
epochs + Phase 4's fusion logic. A suspiciously fast, very high FakeAVCeleb
AUC (like 0.99+ after epoch 1) is a known red flag for models picking up
generator-specific artifacts that don't transfer -- this is the same class
of concern as the leading-silence shortcut found earlier in this project.
If this also fails to generalize to DFDC, better to know now.

Usage (from backend/, with the venv active):
    python eval_dfdc_visual.py "E:\\TruthSeeker-SSD\\DFDC_sample\\train_sample_videos" --limit 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from app.models.visual import FaceManipulationNet
from app.preprocessing.video_utils import extract_lip_sequence

WEIGHTS_PATH = Path("app/models/weights/face_manipulation.pt")


def get_visual_model():
    model = FaceManipulationNet()
    if not WEIGHTS_PATH.exists():
        return model, False
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, True


def score_video(model, video_path: Path, n_frames: int = 8) -> float | None:
    """Returns P(visually manipulated) in [0,1] for one video, or None if
    no usable face crops were found. Sampling matches build_visual_cache's
    even-spacing approach, for a fair comparison against training data."""
    try:
        seq = extract_lip_sequence(video_path)
    except Exception as e:                            # noqa: BLE001
        print(f"  [skip] {video_path.name}: {e}")
        return None

    valid = [f for f in seq if f.face_crop is not None]
    if len(valid) < 2:
        print(f"  [skip] {video_path.name}: fewer than 2 frames with a detected face")
        return None

    idx = np.linspace(0, len(valid) - 1, num=min(n_frames, len(valid)))
    idx = sorted(set(np.round(idx).astype(int).tolist()))
    faces = np.stack([valid[i].face_crop for i in idx]).astype(np.float32)

    with torch.no_grad():
        x = torch.from_numpy(faces).permute(0, 3, 1, 2)  # (n, 3, 224, 224)
        p = model.predict_proba(x)
    return float(p.mean().item())


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Same manual AUC implementation as eval_dfdc.py, kept in sync deliberately."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Folder containing DFDC sample mp4s + metadata.json")
    ap.add_argument("--limit", type=int, default=None)
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

    model, trained = get_visual_model()
    if not trained:
        sys.exit(f"No trained checkpoint found at {WEIGHTS_PATH} "
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
    print(f"DFDC cross-dataset eval — VISUAL detector  "
          f"({len(scores)} videos: {n_real} real, {n_fake} fake; {elapsed:.0f}s)")
    print(f"  accuracy : {accuracy:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print("=" * 50)

    out = {
        "model": "face_manipulation (visual-only)",
        "n_videos": len(scores), "n_real": n_real, "n_fake": n_fake,
        "accuracy": accuracy, "auc": auc,
        "results": [{"file": n, "label": l, "prob": p}
                     for n, l, p in zip(names, labels, scores)],
    }
    with open("dfdc_eval_result_visual.json", "w") as f:
        json.dump(out, f, indent=2)
    print("full results written to dfdc_eval_result_visual.json")


if __name__ == "__main__":
    main()
