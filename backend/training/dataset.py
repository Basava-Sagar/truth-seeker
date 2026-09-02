"""
dataset.py — FakeAVCeleb loader for Truth-Seeker
------------------------------------------------
Expected raw layout (v1.2, as delivered by the authors):

    FakeAVCeleb_v1.2/
        RealVideo-RealAudio/<race>/<gender>/id.../*.mp4     label 0
        RealVideo-FakeAudio/...                             label 1
        FakeVideo-RealAudio/...                             label 1
        FakeVideo-FakeAudio/...                             label 1

Three-stage design:

1. scan()          — walk the tree, extract (path, label, category, identity).
2. build_cache()   — run the Phase 1 pipeline ONCE per video and store the
                     aligned windows as compressed .npz (lips as uint8,
                     mel as float16). Preprocessing (ffmpeg + MediaPipe)
                     dominates cost, so caching gives a ~50x faster epoch.
3. WindowDataset   — serves (lips, mel, label) tensors from the cache.
                     Training mode samples ONE random window per video per
                     epoch (natural augmentation, keeps epochs short);
                     eval mode yields every window for video-level scoring.

Splits are BY IDENTITY (id_ folders), never randomly — the same face in
train and test inflates AUC without measuring generalization.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from app.preprocessing.pipeline import preprocess_video

CATEGORIES = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 1,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
}
_ID_RE = re.compile(r"(id\d+)")


@dataclass
class VideoEntry:
    path: Path
    label: int
    category: str
    identity: str
    cache: Path | None = None   # filled by build_cache


# ----------------------------------------------------------------------
# 1. Scan
# ----------------------------------------------------------------------
def scan(root: str | Path) -> list[VideoEntry]:
    root = Path(root)
    entries: list[VideoEntry] = []
    for cat, label in CATEGORIES.items():
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        for mp4 in cat_dir.rglob("*.mp4"):
            m = _ID_RE.search(str(mp4))
            identity = m.group(1) if m else mp4.parent.name
            entries.append(VideoEntry(mp4, label, cat, identity))
    if not entries:
        raise RuntimeError(
            f"No videos found under {root}. Expected category folders like "
            f"'RealVideo-RealAudio/'. Check the dataset root path.")
    return entries


def split_by_identity(entries: list[VideoEntry], val_frac: float = 0.15,
                      seed: int = 7) -> tuple[list[VideoEntry], list[VideoEntry]]:
    """Deterministic identity-level split — no subject leaks across sets."""
    ids = sorted({e.identity for e in entries})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train = [e for e in entries if e.identity not in val_ids]
    val = [e for e in entries if e.identity in val_ids]
    return train, val


# ----------------------------------------------------------------------
# 2. Preprocessing cache
# ----------------------------------------------------------------------
def _cache_path(cache_dir: Path, video: Path, root: Path | None = None) -> Path:
    # Hash a STABLE key, not the full absolute path. If the dataset root gets
    # mounted at a different absolute location between runs (e.g. Colab's
    # LOCAL_DATA_ROOT corrected from /content/FakeAVCeleb_v1.2 to a nested
    # .../FakeAVCeleb_v1.2/FakeAVCeleb_v1.2), hashing str(video) produces a
    # different filename for the *same* video every time the root changes —
    # silently orphaning the existing cache and forcing a full re-cache.
    # Path relative to the dataset root is stable across sessions/mounts.
    key = str(video.relative_to(root)) if root is not None else video.name
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return cache_dir / f"{video.stem}_{h}.npz"


def _cache_one_video(video_path: Path, out_path: Path, tmp_suffix: str,
                      max_windows_per_video: int) -> tuple[bool, str | None]:
    """
    Worker function run in a separate process: preprocesses ONE video and
    writes its cache .npz atomically. Must be a module-level function (not
    a closure/method) so it can be pickled and sent to worker processes —
    this matters especially on Windows, which uses spawn (not fork) to
    start subprocesses, so closures over local state silently fail there.

    Returns (success, error_message).
    """
    try:
        pv = preprocess_video(video_path)
        wins = pv.windows[:max_windows_per_video]
        lips_u8 = (np.stack([w.lips for w in wins]) * 255.0).clip(0, 255).astype(np.uint8)
        tmp_out = out_path.with_name(out_path.name + tmp_suffix)
        with open(tmp_out, "wb") as f:
            np.savez_compressed(
                f,
                lips=lips_u8,
                mel=np.stack([w.mel for w in wins]).astype(np.float16),
            )
        os.replace(tmp_out, out_path)
        return True, None
    except Exception as exc:                          # noqa: BLE001
        return False, str(exc)


def build_cache(entries: list[VideoEntry], cache_dir: str | Path,
                max_windows_per_video: int = 8, data_root: str | Path | None = None,
                num_workers: int | None = None) -> list[VideoEntry]:
    """
    Preprocesses every video once. Failures (no face, silent, corrupt —
    a handful exist in FakeAVCeleb) are logged and dropped, not fatal.
    Returns only the entries that produced usable windows.

    data_root: pass the dataset root (e.g. args.data) so cache filenames are
    hashed from a path *relative* to it — stable across sessions even if the
    root gets mounted at a different absolute location (e.g. Colab reconnects
    with LOCAL_DATA_ROOT corrected to a nested folder). Without this, a
    changed root silently orphans any existing cache and forces a full redo.

    num_workers: number of processes to preprocess videos in parallel.
    Preprocessing is CPU-bound (ffmpeg + MediaPipe), so this is the single
    biggest speedup available on a machine with an otherwise-idle GPU and
    several CPU cores sitting unused. Defaults to os.cpu_count() (leaving
    none spare) since this step doesn't compete with GPU work. Pass 1 to
    fall back to sequential processing (e.g. for debugging a failure).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = Path(data_root) if data_root is not None else None
    kept, failed = [], []

    # Separate already-cached entries (cheap, sequential) from work that
    # actually needs preprocessing (expensive, parallelizable).
    to_process: list[tuple[VideoEntry, Path]] = []
    for e in entries:
        out = _cache_path(cache_dir, e.path, root)
        if out.exists():
            e.cache = out
            kept.append(e)
        else:
            to_process.append((e, out))

    if not to_process:
        return kept

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, min(num_workers, len(to_process)))

    print(f"  {len(kept)} already cached, {len(to_process)} to process "
          f"using {num_workers} worker process(es)", flush=True)

    done = 0
    if num_workers == 1:
        # Sequential fallback — same behavior as before, useful for
        # debugging a specific failure without multiprocessing noise.
        for e, out in to_process:
            ok, err = _cache_one_video(e.path, out, f".tmp{os.getpid()}", max_windows_per_video)
            done += 1
            if ok:
                e.cache = out
                kept.append(e)
            else:
                failed.append((str(e.path), err))
            if done % 100 == 0:
                print(f"  cached {done}/{len(to_process)} ({len(failed)} failed)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_cache_one_video, e.path, out, f".tmp{i}", max_windows_per_video): (e, out)
                for i, (e, out) in enumerate(to_process)
            }
            for fut in as_completed(futures):
                e, out = futures[fut]
                ok, err = fut.result()
                done += 1
                if ok:
                    e.cache = out
                    kept.append(e)
                else:
                    failed.append((str(e.path), err))
                if done % 100 == 0:
                    print(f"  cached {done}/{len(to_process)} ({len(failed)} failed)", flush=True)

    if failed:
        (cache_dir / "failures.json").write_text(json.dumps(failed, indent=2))
        print(f"  {len(failed)} videos skipped — see failures.json")
    return kept


# ----------------------------------------------------------------------
# 3. Dataset
# ----------------------------------------------------------------------
class WindowDataset(Dataset):
    """
    mode="train": one random window per video per __getitem__ call,
                  with light augmentation (horizontal flip, small
                  brightness jitter on lips; gain jitter on mel).
    mode="eval":  index maps to (video, window) pairs — every window,
                  deterministic, so video-level metrics can aggregate.
    """

    def __init__(self, entries: list[VideoEntry], mode: str = "train"):
        assert mode in ("train", "eval")
        self.entries = entries
        self.mode = mode
        if mode == "eval":
            self.index: list[tuple[int, int]] = []
            for vi, e in enumerate(entries):
                with np.load(e.cache) as z:
                    n = z["lips"].shape[0]
                self.index.extend((vi, wi) for wi in range(n))

    def __len__(self):
        return len(self.entries) if self.mode == "train" else len(self.index)

    def sample_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for WeightedRandomSampler (train)."""
        labels = np.array([e.label for e in self.entries])
        n_real, n_fake = (labels == 0).sum(), (labels == 1).sum()
        w = np.where(labels == 0, 1.0 / max(n_real, 1), 1.0 / max(n_fake, 1))
        return torch.as_tensor(w, dtype=torch.double)

    def _augment(self, lips: np.ndarray, mel: np.ndarray):
        if random.random() < 0.5:
            lips = lips[..., ::-1].copy()                    # horizontal flip
        lips = np.clip(lips * random.uniform(0.85, 1.15), 0, 1)
        mel = mel + np.float32(random.uniform(-0.1, 0.1))    # gain jitter
        return lips, mel

    def __getitem__(self, idx: int):
        if self.mode == "train":
            e = self.entries[idx]
            with np.load(e.cache) as z:
                wi = random.randrange(z["lips"].shape[0])
                lips = z["lips"][wi].astype(np.float32) / 255.0
                mel = z["mel"][wi].astype(np.float32)
            lips, mel = self._augment(lips, mel)
            video_idx = idx
        else:
            video_idx, wi = self.index[idx]
            e = self.entries[video_idx]
            with np.load(e.cache) as z:
                lips = z["lips"][wi].astype(np.float32) / 255.0
                mel = z["mel"][wi].astype(np.float32)

        return (torch.from_numpy(lips), torch.from_numpy(mel),
                torch.tensor(float(e.label)), video_idx)
