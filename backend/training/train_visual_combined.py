"""
train_visual_combined.py — trains FaceManipulationNet on FakeAVCeleb + DFDC
------------------------------------------------------------------------------
The actual root-cause attempt at fixing cross-dataset generalization, after
two FakeAVCeleb-specific shortcuts were found and patched individually
(leading-silence audio artifact, codec/bitrate visual artifact) without the
model actually generalizing to DFDC. Mixing real DFDC data into TRAINING
(not just evaluation) is the standard fix for this class of problem.

Run from backend/, after both caches exist (Phase 2's cache_visual for
FakeAVCeleb, and this script builds cache_visual_dfdc itself on first run):

    python -m training.train_visual_combined \
        --fakeav-data "E:\\...\\FakeAVCeleb_v1.2" --fakeav-cache "E:\\...\\cache_visual" \
        --dfdc-data "E:\\...\\DFDC_sample\\train_sample_videos" --dfdc-cache "E:\\...\\cache_visual_dfdc" \
        --epochs 15

CRITICAL discipline: DFDC is split into a trainable_pool and a PERMANENT
holdout (see dataset_dfdc.split_dfdc_holdout) BEFORE anything else happens.
Only the trainable_pool is ever cached, sampled, or trained on. The holdout
video list is written to holdout_manifest.json so eval_dfdc_holdout.py can
verify it's evaluating the exact same set that was actually held out --
not a different split it re-derives and might accidentally get wrong.

Trains a FRESH model (not fine-tuned from the FakeAVCeleb-only checkpoint)
-- deliberate: that checkpoint's early layers may have absorbed the two
known shortcuts, and fine-tuning from a compromised starting point risks
carrying that bias forward rather than genuinely relearning from scratch.

Checkpoint saved to app/models/weights/face_manipulation_combined.pt --
a NEW path, distinct from the FakeAVCeleb-only face_manipulation.pt, so
the original (known-working, already-evaluated) checkpoint is never
overwritten and stays available as an immediate fallback.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from app.models.visual import FaceManipulationNet
from training.dataset_visual import FrameDataset, build_visual_cache, scan_visual, split_by_identity
from training.dataset_dfdc import scan_dfdc, split_dfdc_holdout
from training.train_visual import auc_score  # reuse, kept in sync deliberately

WEIGHTS_OUT = Path("app/models/weights/face_manipulation_combined.pt")
HOLDOUT_MANIFEST = Path("holdout_manifest.json")

# DFDC entries carry category "REAL"/"FAKE" (see dataset_dfdc.py); FakeAVCeleb
# entries carry categories like "RealVideo-RealAudio" (see dataset_visual.py).
# These string sets never overlap, so category alone identifies source
# dataset without needing a separate field -- used below for the per-source
# validation breakdown.
def _is_dfdc(entry) -> bool:
    return entry.category in ("REAL", "FAKE")


@torch.no_grad()
def evaluate(model, loader, entries, device) -> dict:
    """Video-level AUC, overall AND broken down by source dataset -- lets
    us see whether combined training preserves FakeAVCeleb performance
    while (hopefully) gaining real DFDC generalization, rather than just
    trading one for the other blindly."""
    model.eval()
    per_video: dict[int, list[float]] = defaultdict(list)
    for faces, _labels, vidx in loader:
        p = torch.sigmoid(model(faces.to(device)))
        for v, prob in zip(vidx.tolist(), p.cpu().tolist()):
            per_video[v].append(prob)

    vids = sorted(per_video)
    scores = np.array([float(np.mean(per_video[v])) for v in vids])
    labels = np.array([entries[v].label for v in vids])
    is_dfdc = np.array([_is_dfdc(entries[v]) for v in vids])

    out = {"overall": auc_score(labels, scores)}
    if is_dfdc.any():
        out["dfdc_val"] = auc_score(labels[is_dfdc], scores[is_dfdc])
    if (~is_dfdc).any():
        out["fakeav_val"] = auc_score(labels[~is_dfdc], scores[~is_dfdc])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakeav-data", required=True)
    ap.add_argument("--fakeav-cache", required=True)
    ap.add_argument("--dfdc-data", required=True)
    ap.add_argument("--dfdc-cache", required=True)
    ap.add_argument("--dfdc-holdout-frac", type=float, default=0.5,
                     help="fraction of DFDC identities permanently held out (default 0.5 -- "
                          "DFDC's sample set is small, so this trades training volume for a "
                          "holdout worth trusting)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # --- FakeAVCeleb (unchanged from train_visual.py) ---
    fakeav_entries = scan_visual(args.fakeav_data)
    fakeav_train, fakeav_val = split_by_identity(fakeav_entries)
    print(f"FakeAVCeleb: {len(fakeav_entries)} scanned -> "
          f"{len(fakeav_train)} train / {len(fakeav_val)} val")
    fakeav_train = build_visual_cache(fakeav_train, args.fakeav_cache, data_root=args.fakeav_data)
    fakeav_val = build_visual_cache(fakeav_val, args.fakeav_cache, data_root=args.fakeav_data)

    # --- DFDC: carve out the PERMANENT holdout FIRST, before anything else
    # touches this data. Only dfdc_trainable ever gets cached or trained on.
    dfdc_entries = scan_dfdc(args.dfdc_data)
    dfdc_trainable, dfdc_holdout = split_dfdc_holdout(dfdc_entries, args.dfdc_holdout_frac)
    print(f"DFDC: {len(dfdc_entries)} scanned -> "
          f"{len(dfdc_trainable)} trainable / {len(dfdc_holdout)} PERMANENT HOLDOUT "
          f"(never touched below)")

    HOLDOUT_MANIFEST.write_text(json.dumps(
        {"files": [str(e.path) for e in dfdc_holdout],
         "labels": [e.label for e in dfdc_holdout]}, indent=2))
    print(f"holdout manifest written to {HOLDOUT_MANIFEST} "
          f"({len(dfdc_holdout)} videos, for eval_dfdc_holdout.py)")

    dfdc_train, dfdc_val = split_by_identity(dfdc_trainable)
    print(f"DFDC trainable pool: {len(dfdc_train)} train / {len(dfdc_val)} val (monitoring only)")
    dfdc_train = build_visual_cache(dfdc_train, args.dfdc_cache, data_root=args.dfdc_data)
    dfdc_val = build_visual_cache(dfdc_val, args.dfdc_cache, data_root=args.dfdc_data)

    # --- Combine ---
    combined_train = fakeav_train + dfdc_train
    combined_val = fakeav_val + dfdc_val
    print(f"combined: {len(combined_train)} train videos / {len(combined_val)} val videos")

    train_ds = FrameDataset(combined_train, augment=True)
    val_ds = FrameDataset(combined_val, augment=False)
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    # Weighted by CLASS LABEL only (sample_weights() looks only at .label,
    # never at dataset origin) -- deliberately NOT weighting by source
    # dataset, so the model has no reason to learn "which dataset is this"
    # as a feature. Same discipline that motivated this whole experiment.
    sampler = WeightedRandomSampler(train_ds.sample_weights(),
                                    num_samples=len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True)

    # --- Model: FRESH weights, not fine-tuned from the FakeAVCeleb-only
    # checkpoint (see module docstring for why) ---
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

        metrics = evaluate(model, val_dl, combined_val, device)
        print(f"epoch {epoch:02d}  overall: {metrics['overall']:.4f}  "
              f"fakeav_val: {metrics.get('fakeav_val', float('nan')):.4f}  "
              f"dfdc_val: {metrics.get('dfdc_val', float('nan')):.4f}")

        if metrics["overall"] > best_auc:
            best_auc = metrics["overall"]
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"  ↳ new best (AUC {best_auc:.4f}) saved to {WEIGHTS_OUT}")

    print(f"done. best combined val AUC: {best_auc:.4f}")
    print(f"Next: run eval_dfdc_holdout.py against {HOLDOUT_MANIFEST} for the real test.")


if __name__ == "__main__":
    main()
