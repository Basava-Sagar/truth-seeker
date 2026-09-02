"""
train_visual.py — trains FaceManipulationNet on FakeAVCeleb (Phase 3)
------------------------------------------------------------------------
Run from backend/, AFTER Phase 2's cache is built:

    python -m training.train_visual --data /path/to/FakeAVCeleb_v1.2 \
                                    --cache /path/to/cache_visual --epochs 15

Separate model, separate checkpoint from the lip-sync TruthSeekerNet --
see app/models/visual.py and training/dataset_visual.py for the full
design rationale (visual-only labels, frame-level classification).

Reports video-level AUC each epoch (averaging each video's sampled-frame
predictions), matching train.py's evaluation philosophy: this detector
will be used on whole videos in practice, so that's the number that
actually matters, not frame-level accuracy.

Best checkpoint (by video-level AUC) is written to:
    app/models/weights/face_manipulation.pt

Training data uses augment=True (randomized JPEG re-compression + resize
jitter per sample -- see dataset_visual.py's _compression_jitter) after
diagnose_visual_shortcut.py confirmed a codec/bitrate shortcut (RealVideo-*
~100% h264 @ ~250-280kbps vs FakeVideo-FakeAudio ~100% mpeg4 @ ~470kbps)
was behind the suspiciously fast epoch-1 AUC (0.9941, climbing to 0.9992 by
epoch 15) alongside near-random DFDC cross-dataset generalization (AUC
0.5222, worse than the 87.8% majority-class baseline). Validation stays
augment=False so eval numbers reflect real, un-jittered input.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from app.models.visual import FaceManipulationNet
from training.dataset_visual import FrameDataset, build_visual_cache, scan_visual, split_by_identity

WEIGHTS_OUT = Path("app/models/weights/face_manipulation.pt")


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney), no sklearn dependency -- same
    implementation as train.py's auc_score, kept in sync deliberately."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, loader, entries, device) -> dict:
    """Video-level scores: mean of each video's sampled-frame probabilities."""
    model.eval()
    per_video: dict[int, list[float]] = defaultdict(list)
    for faces, _labels, vidx in loader:
        p = torch.sigmoid(model(faces.to(device)))
        for v, prob in zip(vidx.tolist(), p.cpu().tolist()):
            per_video[v].append(prob)

    vids = sorted(per_video)
    scores = np.array([float(np.mean(per_video[v])) for v in vids])
    labels = np.array([entries[v].label for v in vids])
    return {"overall": auc_score(labels, scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FakeAVCeleb root dir")
    ap.add_argument("--cache", required=True, help="visual-cache dir (from dataset_visual.py)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # --- Data ---
    entries = scan_visual(args.data)
    print(f"scanned {len(entries)} videos "
          f"({sum(e.label == 0 for e in entries)} visually real / "
          f"{sum(e.label == 1 for e in entries)} visually fake)")
    train_e, val_e = split_by_identity(entries)

    print("building/validating visual cache…")
    train_e = build_visual_cache(train_e, args.cache, data_root=args.data)
    val_e = build_visual_cache(val_e, args.cache, data_root=args.data)

    # augment=True for training only -- randomized JPEG re-compression +
    # resize jitter to defeat the codec/bitrate shortcut (see module
    # docstring). Validation stays augment=False so eval numbers reflect
    # real, un-jittered input.
    train_ds = FrameDataset(train_e, augment=True)
    val_ds = FrameDataset(val_e, augment=False)
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    sampler = WeightedRandomSampler(train_ds.sample_weights(),
                                    num_samples=len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True)

    # --- Model / optim ---
    model = FaceManipulationNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_auc = 0.0
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (faces, labels, _vidx) in enumerate(train_dl, 1):
            faces, labels = faces.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=(device == "cuda")):
                logits = model(faces)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {running / step:.4f}", flush=True)
        sched.step()

        metrics = evaluate(model, val_dl, val_e, device)
        print(f"epoch {epoch:02d}  overall: {metrics['overall']:.4f}")

        if metrics["overall"] > best_auc:
            best_auc = metrics["overall"]
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"  ↳ new best (AUC {best_auc:.4f}) saved to {WEIGHTS_OUT}")

    print(f"done. best video-level AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
