"""
pipeline.py
-----------
Combines audio_utils and video_utils into model-ready, temporally
aligned windows. This is the single entry point Phase 2 (model) and
Phase 4 (FastAPI) will call.

Window layout (1-second windows, 50% overlap):
    lips : (T, 1, 96, 96)  with T = 25 frames
    mel  : (1, 80, 100)    100 mel columns = 25 frames x 4 cols/frame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_utils import MEL_COLS_PER_FRAME, video_to_logmel
from .video_utils import VIDEO_FPS, extract_lip_sequence

WINDOW_FRAMES = 25                       # 1 second of video
WINDOW_MELS = WINDOW_FRAMES * MEL_COLS_PER_FRAME   # 100 mel columns
STRIDE_FRAMES = 12                       # ~50% overlap


@dataclass
class AlignedWindow:
    start_time: float                    # seconds
    end_time: float
    lips: np.ndarray                     # (T, 1, 96, 96) float32
    mel: np.ndarray                      # (1, 80, WINDOW_MELS) float32
    face_bboxes: list = field(default_factory=list)  # per-frame, for overlays
    lip_bboxes: list = field(default_factory=list)


@dataclass
class PreprocessedVideo:
    duration: float
    windows: list[AlignedWindow]
    face_detection_rate: float           # fraction of frames with a mouth found
    leading_silence_trimmed: float = 0.0  # seconds cut from start (both modalities)


def preprocess_video(video_path: str | Path) -> PreprocessedVideo:
    """
    mp4 -> aligned (lip, mel) windows.

    Raises RuntimeError with a user-facing message if the video is
    unusable (no audio, no face, or too short), so the API layer can
    return a clean 422 instead of a stack trace.
    """
    video_path = Path(video_path)

    # --- Audio stream ---
    logmel, trimmed_seconds = video_to_logmel(video_path)   # (80, T_mel)

    # --- Video stream ---
    frames = extract_lip_sequence(video_path)     # list[FrameResult] @ 25 fps

    # Drop the same leading duration from video frames that was trimmed
    # from the audio, so modalities stay frame-aligned. See audio_utils
    # .detect_leading_silence for why this trim exists (FakeAVCeleb
    # leading-silence shortcut).
    trim_frames = int(round(trimmed_seconds * VIDEO_FPS))
    frames = frames[trim_frames:]

    # Truncate both modalities to their common duration.
    n_frames = min(len(frames), logmel.shape[1] // MEL_COLS_PER_FRAME)
    frames = frames[:n_frames]

    # Recompute AFTER truncation — counting n_valid on the pre-truncation
    # frame list while dividing by the post-truncation length could push
    # face_detection_rate above 1.0 whenever video frames outnumbered the
    # available mel columns (common, since audio/video durations rarely
    # match exactly, and more so now that the silence-trim can shift them
    # by slightly different frame counts).
    n_valid = sum(1 for f in frames if f.lip_crop is not None)
    if n_frames < WINDOW_FRAMES:
        raise RuntimeError("Video shorter than 1 second — too short to analyze.")
    if n_valid / n_frames < 0.3:
        raise RuntimeError("Could not reliably detect a face — is the speaker visible?")

    windows: list[AlignedWindow] = []
    for start in range(0, n_frames - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        chunk = frames[start:start + WINDOW_FRAMES]
        if any(f.lip_crop is None for f in chunk):
            continue  # skip windows where the face was never found

        lips = np.stack([f.lip_crop for f in chunk])[:, None]   # (T,1,96,96)

        m0 = start * MEL_COLS_PER_FRAME
        mel = logmel[:, m0:m0 + WINDOW_MELS][None]              # (1,80,100)
        if mel.shape[2] < WINDOW_MELS:
            continue

        windows.append(AlignedWindow(
            start_time=chunk[0].timestamp,
            end_time=chunk[-1].timestamp,
            lips=lips.astype(np.float32),
            mel=mel.astype(np.float32),
            face_bboxes=[f.face_bbox for f in chunk],
            lip_bboxes=[f.lip_bbox for f in chunk],
        ))

    if not windows:
        raise RuntimeError("No usable audio-visual windows found in this video.")

    return PreprocessedVideo(
        duration=n_frames / VIDEO_FPS,
        windows=windows,
        face_detection_rate=n_valid / len(frames),
        leading_silence_trimmed=trimmed_seconds,
    )


if __name__ == "__main__":
    import sys

    pv = preprocess_video(sys.argv[1])
    w = pv.windows[0]
    print(f"{pv.duration:.1f}s video -> {len(pv.windows)} windows "
          f"(face rate {pv.face_detection_rate:.0%})")
    print(f"leading silence trimmed: {pv.leading_silence_trimmed*1000:.0f}ms")
    print(f"window 0: lips {w.lips.shape}, mel {w.mel.shape}, "
          f"t=[{w.start_time:.2f}, {w.end_time:.2f}]s")
