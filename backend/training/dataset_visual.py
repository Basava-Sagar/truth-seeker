"""
dataset_visual.py — FakeAVCeleb loader for the visual-manipulation detector
----------------------------------------------------------------------------
Phase 2 of the visual-manipulation detector plan (see truth-seeker project
notes). This is a SEPARATE detector from the lip-sync desync model in
dataset.py / train.py — it looks only at whether the VIDEO itself was
visually manipulated (face-swap / reenactment), independent of whether the
audio is real or fake. That's a deliberately different label scheme from
dataset.py's CATEGORIES:

    RealVideo-RealAudio  -> visual label 0 (genuine video)
    RealVideo-FakeAudio  -> visual label 0 (video itself is NOT manipulated
                             -- only the audio is fake, so this is a REAL
                             example for a visual-only detector, even though
                             dataset.py's audio-visual model labels it 1)
    FakeVideo-RealAudio  -> visual label 1
    FakeVideo-FakeAudio  -> visual label 1

Reuses VideoEntry and split_by_identity from dataset.py — same identity-safe
splitting discipline applies here for exactly the same reason (a face seen
in both train and val inflates accuracy without measuring generalization).

Design differences from dataset.py's build_cache (worth knowing why):
- No audio processing at all -- this detector doesn't use audio, so ffmpeg
  extraction and mel computation are skipped entirely.
- Samples a FIXED NUMBER of individual frames per video, evenly spaced
  across its duration (not overlapping 1-second windows) -- this is a
  frame-level classifier, not a sequence/sync model, so there's no
  windowing math needed, just "give me N representative frames."
- Face crops are much bigger than lip crops (224x224x3 color vs 96x96x1
  grayscale), so cache size is proportionally larger -- see the size
  estimate in build_visual_cache's docstring.

IMPORTANT — torch is NOT imported at module level, on purpose:
build_visual_cache() spawns worker processes (Windows uses "spawn", so each
worker re-imports this entire module from scratch). Caching only needs
numpy/opencv/mediapipe (via video_utils) -- torch is only needed later, by
FrameDataset, for actual training (Phase 3). Importing torch unconditionally
at the top of the file meant every spawned worker paid the cost of loading
torch's heaviest internals (JIT/export/compile subsystems) simultaneously,
competing for the same CPU cores purely for an import none of them needed --
this was the actual cause of the mass worker-startup slowdown/crash, not
just "too many workers." FrameDataset below also no longer inherits from
torch.utils.data.Dataset (DataLoader only needs __len__/__getitem__ --
duck typing, not required inheritance) specifically so the `Dataset` name
doesn't have to be imported at class-definition time either, which runs on
every import regardless of which function ends up being called.

IMPORTANT — compression-augmentation added after epoch-15 diagnosis:
diagnose_visual_shortcut.py confirmed RealVideo-* is ~100% h264 at
~250-280kbps while FakeVideo-FakeAudio is ~100% mpeg4 at ~473kbps -- a
near-perfect trivial separator with nothing to do with actual face-swap
content. This fully explained the suspicious epoch-1 FakeAVCeleb AUC
(0.9941, climbing to 0.9992 by epoch 15) alongside near-random DFDC
cross-dataset generalization (AUC 0.5222, accuracy WORSE than the 87.8%
majority-class baseline). FrameDataset now takes an `augment` flag; when
True, every sampled frame gets randomized JPEG re-compression + mild
resize jitter at __getitem__ time (the cache on disk is untouched -- this
only affects what the model sees, not what's stored), so no fixed
codec/bitrate fingerprint stays available to exploit. Standard technique
in the forgery-detection literature for this exact confound. Use
augment=True for the training split only; keep augment=False for
validation so eval numbers reflect real, un-jittered input.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from app.preprocessing.video_utils import extract_lip_sequence, FACE_SIZE
from training.dataset import VideoEntry, split_by_identity  # noqa: F401 (re-exported)

VISUAL_CATEGORIES = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 0,   # video itself is genuine -- only audio is fake
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
}


def scan_visual(root: str | Path) -> list[VideoEntry]:
    """Same tree walk as dataset.scan(), but with visual-only labels."""
    import re
    root = Path(root)
    id_re = re.compile(r"(id\d+)")
    entries: list[VideoEntry] = []
    for cat, label in VISUAL_CATEGORIES.items():
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        for mp4 in cat_dir.rglob("*.mp4"):
            m = id_re.search(str(mp4))
            identity = m.group(1) if m else mp4.parent.name
            entries.append(VideoEntry(mp4, label, cat, identity))
    if not entries:
        raise RuntimeError(
            f"No videos found under {root}. Expected category folders like "
            f"'RealVideo-RealAudio/'. Check the dataset root path.")
    return entries


def _cache_path(cache_dir: Path, video_path: Path, root: Path | None) -> Path:
    import hashlib
    key = str(video_path.relative_to(root)) if root else str(video_path)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return cache_dir / f"{h}.npz"


def _cache_one_video(video_path: Path, out_path: Path, tmp_suffix: str,
                      n_frames: int) -> tuple[bool, str | None]:
    """
    Worker function (module-level for Windows spawn-picklability, same
    reasoning as dataset.py's _cache_one_video): extract n_frames evenly
    spaced face crops from one video, save as a compressed .npz.

    Deliberately touches only numpy/os/video_utils -- no torch. See the
    module docstring for why that matters on Windows.
    """
    try:
        seq = extract_lip_sequence(video_path)
        valid = [f for f in seq if f.face_crop is not None]
        if len(valid) < 2:
            return False, "fewer than 2 frames with a detected face"

        idx = np.linspace(0, len(valid) - 1, num=min(n_frames, len(valid)))
        idx = np.round(idx).astype(int)
        idx = sorted(set(idx.tolist()))  # dedupe if video is short

        faces = np.stack([valid[i].face_crop for i in idx])       # (n, 224, 224, 3) float32 [0,1]
        assert faces.shape[1:] == (FACE_SIZE, FACE_SIZE, 3), \
            f"unexpected face crop shape {faces.shape[1:]}, expected ({FACE_SIZE},{FACE_SIZE},3)"
        faces_u8 = (faces * 255.0).clip(0, 255).astype(np.uint8)

        tmp_out = out_path.with_name(out_path.name + tmp_suffix)
        with open(tmp_out, "wb") as f:
            np.savez_compressed(f, faces=faces_u8)
        os.replace(tmp_out, out_path)
        return True, None
    except Exception as exc:                          # noqa: BLE001
        return False, str(exc)


def build_visual_cache(entries: list[VideoEntry], cache_dir: str | Path,
                       n_frames_per_video: int = 8, data_root: str | Path | None = None,
                       num_workers: int | None = None) -> list[VideoEntry]:
    """
    Preprocesses every video once, caching n_frames_per_video face crops
    each. Same multiprocessing/atomic-write/resume pattern as dataset.py's
    build_cache -- see that function's docstring for the reasoning (Windows
    spawn-picklability, atomic .tmp-then-rename writes, resumability).

    Cache size estimate: 224x224x3 uint8 per frame compresses to roughly
    15-25KB (photographic content, unlike the near-binary lip crops, so
    compression ratio is lower). At 8 frames/video x ~21,500 videos, expect
    roughly 15-25GB total -- confirm you have room before running on the
    full dataset (this is separate from, and in addition to, the existing
    lip-sync model's ~20GB cache).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = Path(data_root) if data_root is not None else None
    kept, failed = [], []

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
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_cache_one_video, e.path, out, f".tmp{i}", n_frames_per_video): (e, out)
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
        (cache_dir / "failures_visual.json").write_text(json.dumps(failed, indent=2))
        print(f"  {len(failed)} videos skipped — see failures_visual.json")
    return kept


def _compression_jitter(img_u8: np.ndarray) -> np.ndarray:
    """
    Randomly re-compresses/resizes a face crop to scrub away codec- and
    bitrate-specific fingerprints (see diagnose_visual_shortcut.py --
    FakeAVCeleb's RealVideo-* is ~100% h264 at ~250-280kbps while
    FakeVideo-FakeAudio is ~100% mpeg4 at ~470kbps, a near-perfect trivial
    separator that has nothing to do with actual face-swap content). This
    is the standard "compression augmentation" technique used in the
    forgery-detection literature for exactly this confound: by randomizing
    JPEG quality and adding mild resize jitter every time an image is
    sampled, no single fixed compression signature stays available for the
    model to key off, forcing it toward genuine content-level cues instead.

    Applied only when augment=True (training), never during eval/cache
    building, so it doesn't affect what's actually stored on disk -- only
    what the model sees per-epoch, so the same cached frame looks slightly
    different each time it's sampled.
    """
    import cv2
    h, w = img_u8.shape[:2]
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    quality = np.random.randint(35, 96)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if ok:
        bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.6, 0.95)
        small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA)
        bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FrameDataset:
    """
    Serves (face_crop, label, video_index) tensors from the visual cache.
    Unlike dataset.py's WindowDataset (one random window per video per
    epoch, for a sequence model), this is a plain frame-level classifier --
    every cached frame across every video is one independent training
    example. A given video's identity split is still respected (train/val
    entries never share an identity), but within a split all frames of all
    its videos are flattened into one big list of samples.

    video_index indexes into self.entries (the VideoEntry list passed to
    __init__) -- used at eval time to average a video's frame-level
    predictions into one video-level score, matching train.py's evaluate()
    philosophy: this detector will be used on whole videos in practice, so
    video-level AUC is the number that actually matters, not frame-level.

    augment: when True, applies randomized JPEG-recompression + resize
    jitter per sample (see _compression_jitter) to defeat the codec/bitrate
    shortcut found after epoch 15 -- pass True for the training split only,
    False for validation so eval reflects real, un-jittered input.

    Deliberately does NOT inherit from torch.utils.data.Dataset -- PyTorch's
    DataLoader only requires __len__/__getitem__ (map-style dataset duck
    typing), inheritance is a convenience, not a requirement. Skipping it
    means this module never needs `from torch.utils.data import Dataset` at
    the top of the file -- see the module docstring for why that matters
    for the caching workers above, which never touch this class at all but
    would still pay for the import if it were unconditional.
    """

    def __init__(self, entries: list[VideoEntry], progress_every: int = 2000,
                augment: bool = False):
        self.entries = entries
        self.augment = augment
        self.samples: list[tuple[Path, int, int, int]] = []  # (cache_path, frame_idx, label, vidx)

        # Reading each cached .npz's frame count is I/O-bound (open, read
        # header, close -- not CPU work), so a thread pool genuinely helps
        # here despite the GIL: threads release it during actual disk I/O.
        # This is the main training process, not a spawned cache-building
        # worker, so there's no Windows-spawn re-import concern (unlike
        # build_visual_cache's ProcessPoolExecutor) -- plain threads are
        # the right, simpler tool for this specific job.
        from concurrent.futures import ThreadPoolExecutor

        valid_entries = [(vidx, e) for vidx, e in enumerate(entries)
                         if e.cache is not None and e.cache.exists()]

        def _read_count(item):
            vidx, e = item
            with np.load(e.cache) as data:
                return vidx, e, data["faces"].shape[0]

        done = 0
        print(f"  indexing {len(valid_entries)} cached videos…", flush=True)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for vidx, e, n in pool.map(_read_count, valid_entries):
                for i in range(n):
                    self.samples.append((e.cache, i, e.label, vidx))
                done += 1
                if done % progress_every == 0:
                    print(f"  indexed {done}/{len(valid_entries)}", flush=True)
        print(f"  indexed {done}/{len(valid_entries)} — {len(self.samples)} total frame samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def sample_weights(self):
        """Inverse-frequency weights for WeightedRandomSampler (train) --
        same convention as dataset.py's WindowDataset.sample_weights()."""
        import torch
        labels = np.array([s[2] for s in self.samples])
        n_real, n_fake = (labels == 0).sum(), (labels == 1).sum()
        w = np.where(labels == 0, 1.0 / max(n_real, 1), 1.0 / max(n_fake, 1))
        return torch.as_tensor(w, dtype=torch.double)

    def __getitem__(self, idx):
        import torch  # local: only needed at actual training time, never
                       # during the spawned caching workers above, which
                       # never instantiate or call this class at all.
        cache_path, frame_idx, label, vidx = self.samples[idx]
        with np.load(cache_path) as data:
            face_u8 = data["faces"][frame_idx]  # uint8, (H, W, 3)

        if self.augment:
            face_u8 = _compression_jitter(face_u8)

        face = face_u8.astype(np.float32) / 255.0
        face = torch.from_numpy(face).permute(2, 0, 1)  # (3, H, W)
        return face, torch.tensor(label, dtype=torch.float32), vidx


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="FakeAVCeleb root dir")
    ap.add_argument("--cache", required=True, help="visual-cache output dir")
    ap.add_argument("--frames", type=int, default=8,
                     help="frames sampled per video (default: 8)")
    ap.add_argument("--workers", type=int, default=None,
                     help="parallel worker processes (default: all CPU cores)")
    args = ap.parse_args()

    entries = scan_visual(args.data)
    n_real = sum(1 for e in entries if e.label == 0)
    n_fake = sum(1 for e in entries if e.label == 1)
    print(f"scanned {len(entries)} videos ({n_real} visually real / {n_fake} visually fake)")

    print("building/validating visual-manipulation cache…")
    kept = build_visual_cache(entries, args.cache, n_frames_per_video=args.frames,
                               data_root=args.data, num_workers=args.workers)
    print(f"cache complete. {len(kept)}/{len(entries)} videos usable.")
