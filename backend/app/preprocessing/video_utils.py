"""
video_utils.py
--------------
Extracts video frames at a fixed 25 fps and crops a normalized
96x96 grayscale mouth region using MediaPipe FaceLandmarker.

NOTE: this used to use the legacy `mp.solutions.face_mesh.FaceMesh` API.
That API was removed from MediaPipe pip wheels for current Python versions
(only available up to 0.10.9, which has no wheel for Python 3.11+), so this
now uses the modern MediaPipe Tasks API (`FaceLandmarker`) instead. The
underlying face landmark topology (468 points, same indices) is unchanged,
so LIP_LANDMARKS below did not need to change.

Alignment contract (shared with audio_utils.py):
    VIDEO_FPS = 25  -> frame i occurs at t = i / 25 seconds.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

VIDEO_FPS = 25
LIP_SIZE = 96          # output mouth crop is LIP_SIZE x LIP_SIZE
LIP_MARGIN = 0.35      # extra margin around the lip bounding box
FACE_SIZE = 224        # output full-face crop is FACE_SIZE x FACE_SIZE (Phase 1:
                        # visual-manipulation detector, separate from the lip-sync
                        # model). Color, not grayscale -- face-swap blend-boundary
                        # artifacts are often chromatic, which grayscale would discard.
FACE_MARGIN = 0.25     # extra margin around the raw landmark-derived face bbox

# Same canonical face-mesh topology as the legacy API — indices are unchanged.
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,   # outer lower
    185, 40, 39, 37, 0, 267, 269, 270, 409,               # outer upper
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,     # inner lower
    191, 80, 81, 82, 13, 312, 311, 310, 415,              # inner upper
]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "_models" / "face_landmarker.task"


def _ensure_model() -> Path:
    """Downloads the FaceLandmarker model once, caches it locally."""
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


@dataclass
class FrameResult:
    """One processed video frame."""
    timestamp: float                 # seconds from video start
    lip_crop: np.ndarray | None      # (LIP_SIZE, LIP_SIZE) float32 in [0,1], None if no face
    face_bbox: tuple | None          # (x1, y1, x2, y2) in original pixels, for overlays
    lip_bbox: tuple | None           # (x1, y1, x2, y2) in original pixels, for overlays
    face_crop: np.ndarray | None = None  # (FACE_SIZE, FACE_SIZE, 3) float32 RGB in
                                          # [0,1], None if no face. Phase 1 addition for
                                          # the visual-manipulation detector -- optional
                                          # field so existing lip-sync code (which never
                                          # reads it) is completely unaffected.


def extract_frames(video_path: str | Path, target_fps: int = VIDEO_FPS):
    """
    Generator yielding (timestamp_seconds, BGR frame) resampled to target_fps.

    Works by tracking the source timestamps and emitting the nearest source
    frame each time the 1/target_fps clock ticks — robust to variable-fps
    phone recordings, which naive "take every Nth frame" logic is not.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = 1.0 / target_fps
    next_emit_t = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        src_t = frame_idx / src_fps
        # Emit this frame for every output tick it covers.
        while src_t >= next_emit_t - (0.5 / src_fps):
            yield next_emit_t, frame
            next_emit_t += step
        frame_idx += 1

    cap.release()


_extractor_singleton: "LipExtractor | None" = None


def _get_extractor() -> "LipExtractor":
    """
    Returns a process-wide LipExtractor, building it only once.

    Constructing FaceLandmarker (loading the model graph + setting up the
    XNNPACK delegate) is expensive — a few seconds. build_cache() calls this
    once per video across tens of thousands of videos, so creating a fresh
    instance each time was the dominant cost, not actual frame inference.
    Reusing one instance for the whole process avoids paying that setup cost
    more than once. Safe across videos because process_frame's internal
    timestamp counter only needs to keep increasing, never reset — it's an
    internal bookkeeping value for the Tasks API's VIDEO mode, not the
    per-video timestamp used for windowing (that's tracked separately, per
    video, by extract_frames/extract_lip_sequence below).
    """
    global _extractor_singleton
    if _extractor_singleton is None:
        _extractor_singleton = LipExtractor()
    return _extractor_singleton


class LipExtractor:
    """
    Wraps MediaPipe FaceLandmarker to produce normalized mouth crops.
    Create once and reuse — model load is expensive.
    """

    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_ensure_model())),
            running_mode=RunningMode.VIDEO,   # temporal tracking, like the old video-mode FaceMesh
            num_faces=1,
            # Relaxed from 0.5: DFDC cross-dataset testing surfaced real
            # missed detections on small/distant faces (wide framing, face
            # occupies a small fraction of the frame) that were otherwise
            # perfectly detectable -- just below the stricter default. 0.35
            # still rejects genuinely faceless/ambiguous frames.
            min_face_detection_confidence=0.35,
            min_tracking_confidence=0.35,
            min_face_presence_confidence=0.35,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def close(self):
        self._landmarker.close()

    def _detect(self, timestamp: float, frame_bgr: np.ndarray):
        """Runs FaceLandmarker once on a BGR frame, returns raw landmarks or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int(timestamp * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        return result.face_landmarks[0] if result.face_landmarks else None

    @staticmethod
    def _enhance_contrast(frame_bgr: np.ndarray) -> np.ndarray:
        """
        CLAHE (adaptive histogram equalization) on the luminance channel.
        Boosts local contrast in dark/backlit regions without blowing out
        already-bright areas -- unlike a global brightness/gamma bump, this
        targets exactly the failure mode found in DFDC's backlit clips
        (silhouetted face, near-zero contrast) without altering frames that
        already detect fine.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # clipLimit=8.0 (vs CLAHE's typical default of ~2-3): tested against
        # a synthetic near-silhouette region and a higher clip limit was
        # needed to recover meaningfully more contrast (std 1.4 -> 4.5 in
        # that test). Not independently verified against a real backlit
        # detection failure -- worth confirming with debug_clahe.py against
        # an actual failing video before trusting this on the full dataset.
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def process_frame(self, timestamp: float, frame_bgr: np.ndarray) -> FrameResult:
        h, w = frame_bgr.shape[:2]

        lms = self._detect(timestamp, frame_bgr)
        if lms is None:
            # Retry once with contrast-enhanced frame -- catches backlit /
            # underexposed faces the raw frame didn't have enough contrast
            # for. Only paid on failure, so this never slows down or changes
            # behavior on frames that already detect fine.
            lms = self._detect(timestamp, self._enhance_contrast(frame_bgr))

        if lms is None:
            return FrameResult(timestamp, None, None, None)

        # Face bbox (for frontend overlay later)
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        face_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        # Full-face crop for the visual-manipulation detector (Phase 1).
        # Computed independently of the lip crop below -- a degenerate lip
        # region (rare, mouth partly out of frame) shouldn't block an
        # otherwise perfectly good face crop, since this detector doesn't
        # need mouth visibility at all.
        fx1, fy1, fx2, fy2 = face_bbox
        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        fhalf = max(fx2 - fx1, fy2 - fy1) * (0.5 + FACE_MARGIN)
        fx1c, fy1c = int(max(fcx - fhalf, 0)), int(max(fcy - fhalf, 0))
        fx2c, fy2c = int(min(fcx + fhalf, w)), int(min(fcy + fhalf, h))
        face_crop = None
        if fx2c - fx1c >= 16 and fy2c - fy1c >= 16:
            fcrop = frame_bgr[fy1c:fy2c, fx1c:fx2c]
            fcrop = cv2.cvtColor(fcrop, cv2.COLOR_BGR2RGB)
            fcrop = cv2.resize(fcrop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
            face_crop = fcrop.astype(np.float32) / 255.0

        # Lip bbox with margin
        lx = [lms[i].x * w for i in LIP_LANDMARKS]
        ly = [lms[i].y * h for i in LIP_LANDMARKS]
        cx, cy = (min(lx) + max(lx)) / 2, (min(ly) + max(ly)) / 2
        half = max(max(lx) - min(lx), max(ly) - min(ly)) * (0.5 + LIP_MARGIN)
        x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x2, y2 = int(min(cx + half, w)), int(min(cy + half, h))

        if x2 - x1 < 8 or y2 - y1 < 8:  # degenerate crop
            return FrameResult(timestamp, None, face_bbox, None, face_crop)

        crop = frame_bgr[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (LIP_SIZE, LIP_SIZE), interpolation=cv2.INTER_AREA)
        crop = crop.astype(np.float32) / 255.0

        return FrameResult(timestamp, crop, face_bbox, (x1, y1, x2, y2), face_crop)


def extract_lip_sequence(video_path: str | Path) -> list[FrameResult]:
    """
    Full pipeline for one video: 25 fps resampling -> lip crops (+ face
    crops for the Phase 1 visual-manipulation detector).
    Missing-face frames are filled by repeating the previous valid crop
    (keeps the tensor dense; the metadata still records the gap). Lip and
    face crops are gap-filled independently, since one can be degenerate
    while the other is still valid (see process_frame).

    Uses the shared process-wide extractor (see _get_extractor) rather than
    building a new one per video — see that function's docstring for why.
    """
    extractor = _get_extractor()
    results: list[FrameResult] = []
    last_valid_lip: np.ndarray | None = None
    last_valid_face: np.ndarray | None = None
    for t, frame in extract_frames(video_path):
        fr = extractor.process_frame(t, frame)
        lip_crop = fr.lip_crop
        face_crop = fr.face_crop
        if lip_crop is None and last_valid_lip is not None:
            lip_crop = last_valid_lip.copy()
        elif lip_crop is not None:
            last_valid_lip = lip_crop
        if face_crop is None and last_valid_face is not None:
            face_crop = last_valid_face.copy()
        elif face_crop is not None:
            last_valid_face = face_crop
        if lip_crop is not fr.lip_crop or face_crop is not fr.face_crop:
            fr = FrameResult(t, lip_crop, fr.face_bbox, fr.lip_bbox, face_crop)
        results.append(fr)
    return results


if __name__ == "__main__":
    import sys

    seq = extract_lip_sequence(sys.argv[1])
    valid = sum(1 for r in seq if r.lip_crop is not None)
    print(f"{len(seq)} frames @ {VIDEO_FPS} fps, {valid} with a detected mouth "
          f"({len(seq) / VIDEO_FPS:.1f}s of video)")

