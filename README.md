---
title: Truth-Seeker
emoji: 🔍
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# Truth-Seeker — Explainable Cross-Modal Deepfake Detection

Detects deepfakes two ways — checking whether lip motion and speech audio
stay synchronized, and separately whether the face itself shows visual
manipulation artifacts (face-swap/reenactment) — then fuses both signals
into one verdict, explaining every result with heatmaps, attention maps,
and a per-frame disagreement timeline.

**Try it live:** _(add your Hugging Face Space URL here once deployed)_

## Quickstart

### Backend (Python 3.10–3.12, ffmpeg required)
    cd backend
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000

### Frontend (Node 18+)
    cd frontend
    npm install
    npm run dev            # http://localhost:5173

## Pipeline

**Lip-sync detector** (original):

    mp4 upload
      → 25 fps frames + MediaPipe lip crops (96×96)     [preprocessing/video_utils.py]
      → 16 kHz audio → 80-band log-mel @ 100 cols/s     [preprocessing/audio_utils.py]
      → aligned 1 s windows (25 frames ↔ 100 mel cols)  [preprocessing/pipeline.py]
      → dual-stream encoders → cross-modal fusion       [models/streams.py, models/fusion.py]
      → Grad-CAM + attention + sync curve               [explain/gradcam.py]

**Visual-manipulation detector** (added after diagnosing why the lip-sync
model alone didn't generalize to DFDC's face-swap-with-untouched-audio
fakes — see project notes for the full story):

    mp4 upload
      → 224×224 full-face crops (8/video)               [preprocessing/video_utils.py]
      → CNN classifier                                   [models/visual.py]
      → Grad-CAM full-face heatmap                       [explain/gradcam_visual.py]

**Fusion** (max-rule, empirically validated against lip-sync-alone on a
DFDC holdout — `eval_fusion_holdout.py`):

    both detectors' probabilities → fused verdict + evidence   [inference_fusion.py, main.py]

## Training (trained weights already included via Git LFS)

**Lip-sync model:** `backend/training/dataset.py` + `train.py`, trained on
FakeAVCeleb. Checkpoint: `backend/app/models/weights/truth_seeker.pt`.

**Visual model:** `backend/training/dataset_visual.py` +
`train_visual_combined.py`, trained on FakeAVCeleb + a held-out-safe slice
of real DFDC data (mixing in genuine cross-dataset examples was the fix
that actually closed the generalization gap, after two narrower shortcuts
— a leading-silence audio artifact, then a codec/bitrate visual artifact —
were found and patched without it transferring). Checkpoint:
`backend/app/models/weights/face_manipulation_combined.pt`.

Neither FakeAVCeleb nor DFDC is included in this repo (see their official
sources) — training requires downloading them locally first.

## Deployment

Single Docker container (see `Dockerfile`): the React frontend is built
and served as static files by the same FastAPI app that exposes the API,
so there's one host, one URL, no CORS setup needed in production.
Configured for Hugging Face Spaces (Docker SDK) via the metadata block at
the top of this file.

**API:**
- `GET /api/health` — liveness + which models are loaded
- `POST /api/analyze?use_fusion=true` — multipart video upload → full
  analysis JSON. `use_fusion=false` bypasses the visual model and returns
  the original lip-sync-only result — a literal runtime fallback, no
  redeploy needed.

## Smoke tests
    python -m app.preprocessing.pipeline path/to/video.mp4   # shapes + windows
    python -m app.models.fusion                              # forward/backward pass
    python -m app.models.visual                               # visual model forward/backward pass

## Known limitations (documented, not hidden)

- The visual detector only ever sees a face crop; the lip-sync detector
  only sees mouth pixels. Neither alone is a general-purpose manipulation
  detector — that's precisely why fusion helps.
- Face detection can fail on severely underexposed/backlit footage (a
  documented, investigated limitation, not a silent failure — such clips
  are reported as unscorable, not guessed on).
- Cross-modal attention diagonality is computed but not loss-supervised,
  so it's an explanatory visualization, not an independently validated
  signal — `sync_distance` (which *is* supervised) is what actually drives
  the lip-sync verdict.
