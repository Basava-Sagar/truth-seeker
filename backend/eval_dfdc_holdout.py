"""
eval_dfdc_holdout.py -- the REAL test of whether combined training helped.

Evaluates the combined-trained checkpoint (face_manipulation_combined.pt)
against the EXACT DFDC videos that were permanently held out during
training (see train_visual_combined.py / holdout_manifest.json) -- never
touched by caching, sampling, or training in any way.

Reads holdout_manifest.json rather than re-deriving the split from
metadata.json, deliberately: re-deriving risks a silent mismatch (wrong
seed, wrong holdout_frac, code drift between scripts) that would produce a
holdout set different from what was actually excluded from training,
invalidating the whole test without any visible error. Reading the exact
manifest the training run wrote removes that risk entirely.

Usage (from backend/, after train_visual_combined.py has completed):
    python eval_dfdc_holdout.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from app.models.visual import FaceManipulationNet
from app.preprocessing.video_utils import extract_lip_sequence

WEIGHTS_PATH = Path("app/models/weights/face_manipulation_combined.pt")
HOLDOUT_MANIFEST = Path("holdout_manifest.json")


def get_model():
    model = FaceManipulationNet()
    if not WEIGHTS_PATH.exists():
        return model, False
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, True


def score_video(model, video_path: Path, n_frames: int = 8) -> float | None:
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
        x = torch.from_numpy(faces).permute(0, 3, 1, 2)
        p = model.predict_proba(x)
    return float(p.mean().item())


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
    files, holdout_labels = manifest["files"], manifest["labels"]
    print(f"holdout set: {len(files)} videos "
          f"({holdout_labels.count(0)} real / {holdout_labels.count(1)} fake) "
          f"-- never touched during training")

    model, trained = get_model()
    if not trained:
        sys.exit(f"No checkpoint at {WEIGHTS_PATH} -- run train_visual_combined.py first.")

    labels, scores, names = [], [], []
    for i, (fpath, label) in enumerate(zip(files, holdout_labels), 1):
        video_path = Path(fpath)
        prob = score_video(model, video_path)
        if prob is None:
            continue
        labels.append(label)
        scores.append(prob)
        names.append(video_path.name)
        verdict = "fake" if prob >= 0.5 else "real"
        correct = "OK" if (prob >= 0.5) == bool(label) else "WRONG"
        print(f"[{i}/{len(files)}] {video_path.name}: true={'FAKE' if label else 'REAL':4s} "
              f"pred={verdict:4s} p={prob:.3f}  {correct}")

    if not scores:
        sys.exit("No holdout videos were successfully scored.")

    labels_arr, scores_arr = np.array(labels), np.array(scores)
    accuracy = float(((scores_arr >= 0.5).astype(int) == labels_arr).mean())
    auc = roc_auc(labels_arr, scores_arr)
    n_real, n_fake = int((labels_arr == 0).sum()), int((labels_arr == 1).sum())

    print()
    print("=" * 55)
    print(f"TRUE holdout eval — combined-trained model  "
          f"({len(scores)} videos: {n_real} real, {n_fake} fake)")
    print(f"  accuracy : {accuracy:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print("=" * 55)
    print()
    print("Compare against:")
    print("  original lip-sync model on DFDC:        AUC 0.6968, 54.6% accuracy")
    print("  FakeAVCeleb-only visual model on DFDC:   AUC 0.5421, 41.9% accuracy")

    with open("holdout_eval_result.json", "w") as f:
        json.dump({"n_videos": len(scores), "n_real": n_real, "n_fake": n_fake,
                   "accuracy": accuracy, "auc": auc,
                   "results": [{"file": n, "label": l, "prob": p}
                               for n, l, p in zip(names, labels, scores)]}, f, indent=2)
    print("\nfull results written to holdout_eval_result.json")


if __name__ == "__main__":
    main()
