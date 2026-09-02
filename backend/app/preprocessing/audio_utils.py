"""
audio_utils.py
--------------
Extracts the audio track from an .mp4 and converts it to log-mel
spectrograms whose time axis is exactly aligned with the video frames.

Alignment contract (shared with video_utils.py and pipeline.py):
    SAMPLE_RATE = 16000 Hz
    HOP_LENGTH  = 160  samples  -> 1 mel column = 10 ms
    VIDEO_FPS   = 25            -> 1 video frame = 40 ms = 4 mel columns
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

# --- Alignment constants (do not change independently) ---
SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 160          # 10 ms per mel column at 16 kHz
WIN_LENGTH = 400          # 25 ms analysis window
N_MELS = 80
MEL_COLS_PER_FRAME = 4    # (SAMPLE_RATE / HOP_LENGTH) / VIDEO_FPS = 100 / 25


def extract_audio_track(video_path: str | Path, out_wav: str | Path | None = None) -> Path:
    """
    Strips the audio track from a video file into a 16 kHz mono WAV
    using ffmpeg. Returns the path to the WAV file.

    Raises RuntimeError if ffmpeg fails (e.g., the video has no audio
    stream — a surprisingly common case worth surfacing to the user,
    since a silent video cannot be sync-checked at all).
    """
    video_path = Path(video_path)
    if out_wav is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # Windows can't unlink a file while its handle is open
        out_wav = Path(tmp_path)
    out_wav = Path(out_wav)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                      # drop video stream
        "-acodec", "pcm_s16le",     # uncompressed 16-bit PCM
        "-ar", str(SAMPLE_RATE),    # resample to 16 kHz
        "-ac", "1",                 # mono
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path.name}: "
            f"{result.stderr[-500:]}"
        )
    return out_wav


def load_waveform(wav_path: str | Path) -> np.ndarray:
    """Loads a WAV file as a float32 mono waveform at SAMPLE_RATE."""
    y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def detect_leading_silence(y: np.ndarray, top_db: float = 20.0) -> float:
    """
    Returns the duration (seconds) of silence at the very start of the
    waveform, using the same silence definition as librosa.effects.trim.

    Why this exists
    ----------------
    FakeAVCeleb has a well-documented dataset bias: SV2TTS-generated
    fake-audio clips carry ~25-30ms of leading silence that real clips
    don't have (Boldisor et al., "Circumventing shortcuts in
    audio-visual deepfake detection datasets", 2024). A classifier can
    reach >98% accuracy on this artifact alone, without learning any
    real audio-visual sync cue -- so any per-category AUC we report
    (especially FakeVideo-RealAudio / audio-fake categories) is not
    trustworthy unless this leading silence is removed identically
    from every clip, real or fake, before feature extraction.

    top_db=20 matches the threshold used in the paper's own analysis.

    frame_length/hop_length are deliberately much smaller than
    librosa's defaults (2048/512, ~128ms/32ms at 16kHz): the artifact
    is only ~25-30ms long, so the default frame size is too coarse to
    resolve it -- the very first analysis frame straddles the
    silence/speech boundary and never reads as pure silence, silently
    detecting 0ms of silence on every clip (verified empirically).
    160/40 samples (10ms/2.5ms) resolves the boundary correctly.
    """
    non_silent = librosa.effects.split(y, top_db=top_db, frame_length=160, hop_length=40)
    if len(non_silent) == 0:
        return 0.0  # entire clip is "silent" by this threshold; don't trim
    first_sample = non_silent[0][0]
    return first_sample / SAMPLE_RATE


def waveform_to_logmel(y: np.ndarray) -> np.ndarray:
    """
    Converts a waveform to a log-mel spectrogram.

    Returns
    -------
    logmel : np.ndarray of shape (N_MELS, T_mel)
        Per-utterance mean/variance normalized log-mel spectrogram.
        T_mel ≈ len(y) / HOP_LENGTH, i.e., 100 columns per second.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # Per-clip normalization stabilizes training across recording conditions.
    mean, std = logmel.mean(), logmel.std()
    logmel = (logmel - mean) / (std + 1e-6)
    return logmel.astype(np.float32)


def video_to_logmel(video_path: str | Path, trim_leading_silence: bool = True) -> tuple[np.ndarray, float]:
    """
    Convenience wrapper: mp4 -> aligned log-mel spectrogram.

    Returns (logmel, trimmed_seconds). trimmed_seconds is how much
    leading silence was cut from the start of the waveform -- the
    caller (pipeline.py) MUST drop the same duration from the start
    of the video frame sequence, or audio/video will desync. Pass
    trim_leading_silence=False to disable (e.g. for debugging).
    """
    wav = extract_audio_track(video_path)
    try:
        y = load_waveform(wav)
        trimmed_seconds = 0.0
        if trim_leading_silence:
            trimmed_seconds = detect_leading_silence(y)
            trim_samples = int(round(trimmed_seconds * SAMPLE_RATE))
            y = y[trim_samples:]
        return waveform_to_logmel(y), trimmed_seconds
    finally:
        wav.unlink(missing_ok=True)  # clean up temp file


if __name__ == "__main__":
    import sys

    mel, trimmed = video_to_logmel(sys.argv[1])
    print(f"log-mel shape: {mel.shape}  "
          f"(~{mel.shape[1] / 100:.1f}s of audio, {MEL_COLS_PER_FRAME} cols/frame)")
    print(f"leading silence trimmed: {trimmed*1000:.0f}ms")
