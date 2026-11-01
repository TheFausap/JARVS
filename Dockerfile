# syntax=docker/dockerfile:1

# ==============================================================================
# Stage 1: Build the React + Vite + Tailwind command-center UI
# ==============================================================================
FROM node:22-alpine AS web-builder

WORKDIR /build/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
# vite.config.js outputs to ../jarvs/web_dist -> /build/jarvs/web_dist
RUN npm run build


# ==============================================================================
# Stage 2: Python runtime (stdlib HTTP/SSE server + memory + voice + agents)
# ==============================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    JARVS_DATA_DIR=/app/data \
    JARVS_VOICES_DIR=/app/voices \
    JARVS_SELF_DIR=/app/self \
    JARVS_WEB_HOST=0.0.0.0 \
    JARVS_WEB_PORT=8765

WORKDIR /app

# System packages:
# - alsa-utils / pulseaudio-utils: `aplay` / `paplay` for Piper TTS audio output
# - iproute2: `ip route` used by jarvs/system_sense.py
# - curl: container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        alsa-utils \
        pulseaudio-utils \
        iproute2 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so sentence-transformers doesn't pull ~2.5GB
# of CUDA wheels into the container (JARVS_EMBED_DEVICE defaults to "cpu").
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Pre-populate the default MiniLM embedding model so first boot works offline
# and without startup lag.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code, self-model dossier, and compiled web UI
COPY jarvs/ ./jarvs/
COPY self/ ./self/
COPY --from=web-builder /build/jarvs/web_dist ./jarvs/web_dist

# Ensure mount points exist for persistent data and Piper voice models
RUN mkdir -p /app/data /app/voices

# LLM endpoint configuration (can be set at build time via --build-arg or
# overridden at runtime via -e / docker-compose environment)
ARG JARVS_BASE_URL=http://192.168.100.248:8081/v1
ARG JARVS_MODEL=DavidAU/Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF
ARG JARVS_API_KEY=sk-no-key-required

ENV JARVS_BASE_URL=${JARVS_BASE_URL} \
    JARVS_MODEL=${JARVS_MODEL} \
    JARVS_API_KEY=${JARVS_API_KEY}

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${JARVS_WEB_PORT:-8765}/api/status >/dev/null || exit 1

CMD ["python", "-m", "jarvs.server"]
