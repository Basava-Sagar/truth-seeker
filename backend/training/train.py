"""
train.py — trains TruthSeekerNet on FakeAVCeleb
-----------------------------------------------
Run from backend/:

    # one-time preprocessing cache (hours; resumable — reruns skip done work)
    python -m training.train --data /path/to/FakeAVCeleb_v1.2 \
                             --cache /path/to/cache --prepare-only

    # training
    python -m training.train --data /path/to/FakeAVCeleb_v1.2 \
                             --cache /path/to/cache --epochs 30

Reports overall AND per-category video-level AUC each epoch — a model
can look strong overall while failing on FakeVideo-RealAudio (Wav2Lip),
the category where lip-sync is engineered to look correct.

Best checkpoint (by overall AUC) is written straight to the path the
API loads from: app/models/weights/truth_seeker.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from app.models.fusion import TruthSeekerNet, truth_seeker_loss
from training.dataset import WindowDataset, build_cache, scan, split_by_identity

WEIGHTS_OUT = Path("app/models/weights/truth_seeker.pt")


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann–Whitney), no sklearn dependency."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, loader, entries, device) -> dict:
    """Video-level scores: mean of each video's top-25% window probs
    (mirrors inference.py so training metrics match production)."""
    model.eval()
    per_video: dict[int, list[float]] = defaultdict(list)
    for lips, mel, _labels, vidx in loader:
        p = torch.sigmoid(model(lips.to(device), mel.to(device))[0])
        for v, prob in zip(vidx.tolist(), p.cpu().tolist()):
            per_video[v].append(prob)

    vids = sorted(per_video)
    scores = np.array([
        float(np.sort(per_video[v])[-max(1, round(len(per_video[v]) * .25)):].mean())
        for v in vids])
    labels = np.array([entries[v].label for v in vids])
    cats = np.array([entries[v].category for v in vids])

    out = {"overall": auc_score(labels, scores)}
    real_mask = labels == 0
    for cat in sorted(set(cats)):
        if cat == "RealVideo-RealAudio":
            continue
        mask = (cats == cat) | real_mask       # each fake type vs all reals
        out[cat] = auc_score(labels[mask], scores[mask])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FakeAVCeleb root dir")
    ap.add_argument("--cache", required=True, help="preprocessing cache dir")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # --- Data ---
    entries = scan(args.data)
    print(f"scanned {len(entries)} videos "
          f"({sum(e.label == 0 for e in entries)} real / "
          f"{sum(e.label == 1 for e in entries)} fake)")
    train_e, val_e = split_by_identity(entries)

    print("building/validating preprocessing cache…")
    train_e = build_cache(train_e, args.cache, data_root=args.data)
    val_e = build_cache(val_e, args.cache, data_root=args.data)
    if args.prepare_only:
        print("cache complete.")
        return

    train_ds = WindowDataset(train_e, "train")
    val_ds = WindowDataset(val_e, "eval")
    sampler = WeightedRandomSampler(train_ds.sample_weights(),
                                    num_samples=len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True)

    # --- Model / optim ---
    model = TruthSeekerNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_auc = 0.0
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (lips, mel, labels, _) in enumerate(train_dl, 1):
            lips, mel = lips.to(device), mel.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=(device == "cuda")):
                logits, aux = model(lips, mel)
                loss = truth_seeker_loss(logits, labels, aux)
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
        line = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"epoch {epoch:02d}  {line}")

        if metrics["overall"] > best_auc:
            best_auc = metrics["overall"]
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"  ↳ new best (AUC {best_auc:.4f}) saved to {WEIGHTS_OUT}")

    print(f"done. best video-level AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
