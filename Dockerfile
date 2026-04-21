# ── Build stage ──────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps (TF needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app/

# Copy model files
# IMPORTANT: You must have model/dnn_ddos_model.h5 and model/labels.txt
# committed to your repository (or use a cloud storage URL — see README)
COPY model/ ./model/

# ── Runtime ──────────────────────────────────────────────
EXPOSE 8000

# Render sets PORT env var — honour it
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
