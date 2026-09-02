# syntax=docker/dockerfile:1

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ .
RUN npm run build

# ---- Stage 2: Python backend + built frontend, single runtime image ----
FROM python:3.11-slim

# ffmpeg: required by audio_utils.py's video_to_logmel (audio extraction).
# libgl1/libglib2.0-0: required by opencv-python's native bindings even in
# headless environments -- without these, cv2 import fails at container
# startup with a cryptic libGL.so.1 error.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Pre-download MediaPipe's face landmark model at BUILD time, not at first
# request -- video_utils.py's _ensure_model() downloads it from Google if
# missing. Doing this here means the container is fully self-contained and
# the first real user request isn't slowed down by (or dependent on) an
# outbound download succeeding at runtime.
RUN python -c "from app.preprocessing.video_utils import _ensure_model; _ensure_model()"

# Built frontend goes here -- app/main.py's FRONTEND_DIST points at this
# exact path (backend/static, i.e. /app/static inside the container).
COPY --from=frontend-build /app/frontend/dist ./static

# Hugging Face Spaces sets $PORT; default to 7860 (HF's conventional port)
# for local `docker run` testing without an explicit -e PORT=... flag.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
