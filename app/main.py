"""
AI-Powered DDoS Detection — Cloud API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from contextlib import asynccontextmanager
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
scaler = None
label_names = None

THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MODEL_PATH = os.getenv("MODEL_PATH", "model/dnn_ddos_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "model/labels.txt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, label_names

    import pickle
    import tensorflow as tf

    logger.info("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            label_names = [line.strip() for line in f if line.strip()]

    # ✅ Warm-up (prevents cold delay)
    dummy = np.zeros((1, model.input_shape[-1]))
    model.predict(dummy, verbose=0)

    logger.info("Model loaded successfully")

    yield


app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: list[float]

    @validator("features")
    def not_empty(cls, v):
        if not v:
            raise ValueError("features cannot be empty")
        return v


@app.get("/")
def root():
    return {"message": "DDoS Detection API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "input_dim": int(model.input_shape[-1]),
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    expected = int(model.input_shape[-1])

    if len(req.features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features, got {len(req.features)}"
        )

    x = np.array(req.features).reshape(1, -1)

    if scaler is not None:
        x = scaler.transform(x)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    if label_names and len(label_names) == len(probs):
        label = label_names[idx]
    else:
        label = str(idx)

    return {
        "prediction": label,
        "confidence": confidence,
        "flagged_unknown": confidence < THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )