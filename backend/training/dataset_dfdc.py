"""
dataset_dfdc.py — DFDC loader for combined-dataset training
--------------------------------------------------------------
Purpose: after finding TWO FakeAVCeleb-specific shortcuts (leading-silence
audio artifact, then a codec/bitrate visual artifact) that a
single-dataset-trained model kept exploiting, this mixes real DFDC data
into training -- the actual root-cause fix, not another patch. See
truth-seeker project notes for the full decision history.

CRITICAL discipline, worth stating explicitly: this module enforces a
PERMANENT holdout split before any DFDC data touches training. Evaluating
on DFDC videos that were partly trained on would prove nothing -- see
split_dfdc_holdout(). The holdout is carved out here, once, at the data
layer, rather than trusted to whoever writes the training script later.

DFDC metadata.json format: {filename: {"label": "REAL"|"FAKE",
"split": ..., "original": <source real filename, FAKE entries only>, ...}}.

Identity grouping: DFDC's "original" field links each fake video to the
real video it was generated from. A FAKE and its source REAL are the same
underlying person -- exactly the leakage case split_by_identity exists to
prevent, so we key .identity off the ORIGINAL real filename for both real
and fake entries (a fake and its source share one identity key; unrelated
real videos each get their own).
"""

from __future__ import annotations

import json
from pathlib import Path

from training.dataset import VideoEntry, split_by_identity  # noqa: F401 (re-exported)


def scan_dfdc(data_dir: str | Path) -> list[VideoEntry]:
    """
    data_dir: folder containing DFDC mp4s + metadata.json (e.g. Kaggle's
    extracted train_sample_videos/). Labels visual-fake=1 / visual-real=0,
    matching FaceManipulationNet's convention (dataset_visual.py).
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found in {data_dir}")

    metadata = json.loads(meta_path.read_text())
    entries: list[VideoEntry] = []
    for fname, info in metadata.items():
        video_path = data_dir / fname
        if not video_path.exists():
            continue
        label = 1 if info["label"].upper() == "FAKE" else 0
        original = info.get("original")
        # For a FAKE video, identity = its source real video's filename
        # (stem) -- ties it to that same person/clip. For a REAL video,
        # identity = its own filename stem (each real clip is its own
        # source unless some other fake also points at it, in which case
        # they'll naturally share this same key too).
        identity = Path(original).stem if original else Path(fname).stem
        entries.append(VideoEntry(video_path, label, "REAL" if label == 0 else "FAKE", identity))

    if not entries:
        raise RuntimeError(f"No usable videos found in {data_dir} (check metadata.json / file paths)")
    return entries


def split_dfdc_holdout(entries: list[VideoEntry], holdout_frac: float = 0.5,
                       seed: int = 99) -> tuple[list[VideoEntry], list[VideoEntry]]:
    """
    Splits DFDC entries into (trainable_pool, permanent_holdout), identity-
    safe (a fake and its source real never end up on opposite sides).

    holdout_frac defaults to 0.5, not dataset.py's usual 0.15 -- DFDC's
    sample set is small (our usable subset has been in the dozens of real
    videos, low hundreds of fake), so a generous holdout is needed just to
    get a holdout set large enough for a meaningful final AUC at all. This
    trades away some potential training signal for a holdout worth trusting
    -- worthwhile given the whole point of this experiment is measuring
    genuine generalization, not maximizing training data volume.

    seed=99 is deliberately DIFFERENT from split_by_identity's default
    seed=7 (used later to further split the trainable_pool into its own
    train/val for monitoring) -- using the same seed for both splits could
    otherwise correlate them in ways that aren't obviously wrong but aren't
    obviously right either; separate seeds keeps the two splits independent.

    The trainable_pool is NOT yet a train/val split -- it's the full set of
    DFDC data allowed to touch training at all. Call split_by_identity()
    again on trainable_pool (default seed=7) to get its own train/val for
    monitoring during training, same as FakeAVCeleb's split.
    """
    holdout, trainable_pool = split_by_identity(entries, val_frac=1 - holdout_frac, seed=seed)
    # split_by_identity returns (train, val) with val_frac fraction in val;
    # we want holdout_frac fraction held out, so invert which side is which.
    return trainable_pool, holdout


if __name__ == "__main__":
    import sys

    entries = scan_dfdc(sys.argv[1])
    n_real = sum(1 for e in entries if e.label == 0)
    n_fake = sum(1 for e in entries if e.label == 1)
    n_identities = len({e.identity for e in entries})
    print(f"scanned {len(entries)} videos ({n_real} real / {n_fake} fake) "
          f"across {n_identities} unique identities")

    trainable, holdout = split_dfdc_holdout(entries)
    print(f"trainable pool: {len(trainable)} videos "
          f"({sum(e.label==0 for e in trainable)} real / {sum(e.label==1 for e in trainable)} fake)")
    print(f"permanent holdout: {len(holdout)} videos "
          f"({sum(e.label==0 for e in holdout)} real / {sum(e.label==1 for e in holdout)} fake)")
