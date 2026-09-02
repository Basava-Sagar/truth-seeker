"""
diagnose_visual_shortcut.py -- check for low-level artifact shortcuts
-------------------------------------------------------------------------
A suspiciously high FakeAVCeleb AUC after 1 epoch (0.9941) combined with
near-random DFDC generalization (0.52) suggests FaceManipulationNet may be
keying off generator-specific low-level signal (resolution, compression,
codec) rather than genuine visual manipulation artifacts -- the same class
of bug as the leading-silence audio shortcut found earlier in this project.

This samples N videos from each category and compares resolution, file
size, and duration. A clean split between real/fake categories on any of
these (e.g. all FakeVideo-* at one resolution, all RealVideo-* at another)
would be enough on its own for a CNN to "cheat" without ever learning to
recognize a face-swap.

Usage (from backend/):
    python diagnose_visual_shortcut.py --data "E:\\TruthSeeker-SSD\\FakeAVCeleb_v1.2" --n 40
"""
import argparse
import random
import subprocess
import json
from pathlib import Path
from collections import defaultdict

CATEGORIES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
              "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


def probe(video_path: Path) -> dict | None:
    """ffprobe -> resolution, duration, file size, codec, bitrate."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=15)
        info = json.loads(out.stdout)
        vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
        return {
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "codec": vstream.get("codec_name"),
            "duration": float(info["format"].get("duration", 0)),
            "size_bytes": int(info["format"].get("size", 0)),
            "bitrate": int(info["format"].get("bit_rate", 0)),
        }
    except Exception as e:                             # noqa: BLE001
        print(f"  ffprobe failed on {video_path.name}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40, help="videos sampled per category")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data)
    rng = random.Random(args.seed)
    results = defaultdict(list)

    for cat in CATEGORIES:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            print(f"skipping {cat} (not found)")
            continue
        videos = list(cat_dir.rglob("*.mp4"))
        sample = rng.sample(videos, min(args.n, len(videos)))
        print(f"probing {len(sample)} videos from {cat}...")
        for v in sample:
            info = probe(v)
            if info:
                results[cat].append(info)

    print("\n" + "=" * 70)
    print(f"{'category':22s} {'resolution':>12s} {'avg size(MB)':>13s} "
          f"{'avg dur(s)':>11s} {'avg kbps':>10s} {'codecs seen'}")
    print("-" * 70)
    for cat, rows in results.items():
        if not rows:
            continue
        res_counts = defaultdict(int)
        for r in rows:
            res_counts[f"{r['width']}x{r['height']}"] += 1
        top_res = max(res_counts, key=res_counts.get)
        res_note = top_res if len(res_counts) == 1 else f"{top_res}(+{len(res_counts)-1} more)"
        avg_size = sum(r["size_bytes"] for r in rows) / len(rows) / 1e6
        avg_dur = sum(r["duration"] for r in rows) / len(rows)
        avg_kbps = sum(r["bitrate"] for r in rows) / len(rows) / 1000
        codecs = set(r["codec"] for r in rows)
        print(f"{cat:22s} {res_note:>12s} {avg_size:>13.2f} {avg_dur:>11.2f} "
              f"{avg_kbps:>10.0f} {codecs}")
    print("=" * 70)

    real_cats = ["RealVideo-RealAudio", "RealVideo-FakeAudio"]
    fake_cats = ["FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
    real_res = set(f"{r['width']}x{r['height']}" for c in real_cats for r in results.get(c, []))
    fake_res = set(f"{r['width']}x{r['height']}" for c in fake_cats for r in results.get(c, []))

    print()
    if real_res and fake_res and not (real_res & fake_res):
        print("⚠️  REAL and FAKE categories use COMPLETELY DISJOINT resolutions.")
        print("    This alone is enough for a CNN to trivially separate them")
        print("    without learning anything about face manipulation.")
    elif real_res != fake_res:
        print("⚠️  REAL and FAKE categories show DIFFERENT resolution distributions")
        print("    (not fully disjoint, but not matched either) -- worth a closer look.")
    else:
        print("✅ resolution doesn't look like an obvious shortcut here.")
    print("Also compare avg size(MB)/kbps above by eye -- a consistent gap there")
    print("is just as exploitable as resolution, and CNNs pick up on compression")
    print("artifacts (blocking, ringing) very easily.")


if __name__ == "__main__":
    main()
