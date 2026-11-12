"""
calibrate_threshold.py -- pick the fake/real cutoff from held-out data
----------------------------------------------------------------------
Scores every video in holdout_manifest.json (plus any extra REAL videos
you point it at), then, for each strategy, finds the lowest threshold whose
false-positive rate on REAL videos is <= a target, and reports the true-
positive rate you get at that threshold.

Writes fusion_threshold.json, which app/inference_fusion.py reads at startup.

Usage (from backend/):
    python calibrate_threshold.py                       # score + calibrate
    python calibrate_threshold.py --extra-real C:\\my_real_clips
    python calibrate_threshold.py --from-cache          # reuse holdout_scores.json
    python calibrate_threshold.py --fpr 0.05            # stricter target

--extra-real: folder of videos you KNOW are genuine, ideally including
WhatsApp/phone-recompressed ones. The DFDC holdout has few reals, and none
that look like WhatsApp compression, so this is what teaches the cutoff
about that case.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

MANIFEST = Path("holdout_manifest.json")
CACHE = Path("holdout_scores.json")
OUT = Path("fusion_threshold.json")
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
STRATEGIES = ["lipsync", "visual", "max", "weighted0.5"]


def fuse(strategy, lip, vis):
    if strategy == "lipsync":
        return lip
    if vis is None:
        return lip  # same per-video fallback as fuse_scores()
    if strategy == "visual":
        return vis
    if strategy == "max":
        return max(lip, vis)
    return 0.5 * vis + 0.5 * lip


def pick_threshold(real, fake, target_fpr):
    """Smallest t with FPR(t) <= target, verdict rule being score >= t."""
    for t in np.unique(np.concatenate([real, fake])):
        if (real >= t).mean() <= target_fpr:
            return float(t)
    return 1.0


def score_all(extra_real):
    # heavy imports are lazy so --from-cache works without torch installed
    from eval_fusion_holdout import score_lipsync
    from app.inference_fusion import score_visual

    items = []
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        items += list(zip(m["files"], m["labels"]))
    else:
        print(f"warning: {MANIFEST} not found, using only --extra-real videos")
    if extra_real:
        items += [(str(p), 0) for p in sorted(Path(extra_real).iterdir())
                  if p.suffix.lower() in VIDEO_EXT]
    if not items:
        sys.exit("Nothing to score.")

    rows = []
    for i, (f, label) in enumerate(items, 1):
        lip = score_lipsync(Path(f))
        if lip is None:
            print(f"  skip {Path(f).name}: lip-sync preprocessing failed")
            continue
        rows.append({"file": Path(f).name, "label": int(label),
                     "lipsync": lip, "visual": score_visual(Path(f))})
        if i % 10 == 0:
            print(f"  ...{i}/{len(items)} scored")
    CACHE.write_text(json.dumps(rows, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.10,
                    help="target false-positive rate on real videos")
    ap.add_argument("--extra-real", default=None)
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--strategy", default="max", choices=STRATEGIES,
                    help="which strategy to write to fusion_threshold.json")
    args = ap.parse_args()

    rows = (json.loads(CACHE.read_text()) if args.from_cache
            else score_all(args.extra_real))
    labels = np.array([r["label"] for r in rows])
    n_real, n_fake = int((labels == 0).sum()), int((labels == 1).sum())
    print(f"\n{len(rows)} videos scored: {n_real} real / {n_fake} fake")
    if n_real < 50:
        print(f"WARNING: only {n_real} real videos. At a {args.fpr:.0%} target that "
              f"is ~{n_real * args.fpr:.1f} false positives, so the threshold is "
              f"noisy. Add more with --extra-real.")

    print(f"\n{'strategy':12s} {'thr@0.5':>22s}   {'calibrated thr':>14s} "
          f"{'FPR':>6s} {'TPR':>6s} {'bal.acc':>8s}")
    chosen = {}
    for s in STRATEGIES:
        sc = np.array([fuse(s, r["lipsync"], r["visual"]) for r in rows])
        real, fake = sc[labels == 0], sc[labels == 1]
        fpr0, tpr0 = (real >= .5).mean(), (fake >= .5).mean()
        t = pick_threshold(real, fake, args.fpr)
        fpr, tpr = (real >= t).mean(), (fake >= t).mean()
        print(f"{s:12s} FPR {fpr0:5.1%} TPR {tpr0:5.1%}   {t:14.3f} "
              f"{fpr:6.1%} {tpr:6.1%} {(tpr + 1 - fpr) / 2:8.3f}")
        chosen[s] = {"threshold": round(t, 4), "fpr": round(float(fpr), 4),
                     "tpr": round(float(tpr), 4)}

    pick = chosen[args.strategy]
    # never silently go BELOW 0.5; the point of this is to cut false alarms
    thr = max(0.5, pick["threshold"])
    OUT.write_text(json.dumps({
        "strategy": args.strategy, "threshold": thr,
        "target_fpr": args.fpr, "n_real": n_real, "n_fake": n_fake,
    }, indent=2))
    print(f"\nWrote {OUT}: strategy={args.strategy}, threshold={thr:.3f}")


if __name__ == "__main__":
    main()
