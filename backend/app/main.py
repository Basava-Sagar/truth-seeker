"""
main.py  (Phase 4 — FastAPI application)
----------------------------------------
Run from backend/ with:
    uvicorn app.main:app --reload --port 8000

Endpoints
---------
GET  /api/health          liveness + whether trained weights are loaded
POST /api/analyze         multipart video upload -> full analysis JSON
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .inference import analyze_video, get_model
from .inference_fusion import analyze_video_fused, get_visual_model

MAX_UPLOAD_MB = 100
ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# Built frontend (frontend/dist, produced by `npm run build`) gets copied
# here at Docker build time -- see Dockerfile. In local dev this directory
# won't exist (you run Vite's dev server separately on :5173 instead), so
# every mount/route below checks FRONTEND_DIST.is_dir() first and no-ops
# if it's absent -- local dev workflow is completely unaffected.
FRONTEND_DIST = Path(__file__).parent.parent / "static"

app = FastAPI(
    title="Truth-Seeker API",
    description="Explainable cross-modal deepfake detection",
    version="1.0.0",
)

# Frontend dev server origins. In production, frontend and backend are
# served from the SAME origin (single container -- see Dockerfile), so no
# cross-origin requests happen there at all; these entries only matter for
# local development (Vite dev server on :5173 talking to uvicorn on :8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_model():
    """Load weights at boot so the first request isn't slow."""
    get_model()
    get_visual_model()


@app.get("/api/health")
def health():
    _, lipsync_trained = get_model()
    _, visual_trained = get_visual_model()
    return {
        "status": "ok",
        "model_trained": lipsync_trained,
        "visual_model_trained": visual_trained,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), use_fusion: bool = True):
    """
    use_fusion=True (default): combines the lip-sync and visual detectors
    via max-rule fusion (see eval_fusion_holdout.py -- beat lip-sync-alone
    by +0.09 AUC on a DFDC holdout set that neither model was trained on).

    use_fusion=False: the ORIGINAL, unmodified lip-sync-only path -- exact
    same call the app made before fusion existed. This is the literal
    fallback: pass ?use_fusion=false on the request, no code change or
    redeploy needed, if fusion is ever suspected of hurting results on
    some class of video the holdout set didn't cover.
    """
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Accepted: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    # Stream the upload to a temp file with a size cap.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    size = 0
    try:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            size += len(chunk)
            if size > MAX_UPLOAD_MB * (1 << 20):
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
                )
            tmp.write(chunk)
        tmp.close()

        # Inference is CPU/GPU-bound and synchronous; run it in a
        # worker thread so the event loop stays responsive.
        if use_fusion:
            fused = await anyio.to_thread.run_sync(
                lambda: analyze_video_fused(tmp_path, method="max"))
            # Flatten: start from the ORIGINAL lip-sync explanation dict
            # unchanged (video/windows/sync_timeline/explanation all stay
            # exactly as analyze_video() already produces them -- the
            # heatmaps and attention map are still lip-sync evidence, since
            # the visual model doesn't have its own explanation UI yet).
            # Only the headline verdict/probability are overridden with the
            # fused result, which is what actually decides "fake" or "real".
            result = dict(fused["lipsync"])
            result["lipsync_probability"] = fused["lipsync"]["fake_probability"]
            result["visual_probability"] = fused["visual_probability"]
            result["visual_heatmap_b64"] = fused["visual_heatmap_b64"]
            result["visual_heatmap_time"] = fused["visual_heatmap_time"]
            result["fake_probability"] = round(fused["fused_probability"], 4)
            result["verdict"] = fused["fused_verdict"]
            result["fusion_method"] = fused["fusion_method"]
        else:
            # Literal fallback: the original, unmodified lip-sync-only path.
            result = await anyio.to_thread.run_sync(analyze_video, tmp_path)
        return result

    except HTTPException:
        raise
    except RuntimeError as exc:
        # User-facing preprocessing errors (no audio, no face, too short).
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed unexpectedly.")
    finally:
        tmp.close()
        tmp_path.unlink(missing_ok=True)


# --- Serve the built frontend, if present (production/Docker only) ---
# Mounted AFTER every /api/* route above, so those always take priority --
# StaticFiles never intercepts an /api/ request. The catch-all below
# handles client-side routing (React Router-style paths that aren't real
# files) by falling back to index.html, letting the SPA's own router take
# over -- standard pattern for single-container SPA + API deployments.
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


# Optional: `python -m app.main` convenience runner.
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # HF Spaces sets $PORT
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
