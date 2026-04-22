import os
import pickle
import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "dnn_ddos_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")
EXPECTED_FEATURES = 52

model  = None
scaler = None
labels = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, labels

    try:
        import tensorflow as tf
        logger.info(f"TensorFlow {tf.__version__}")
    except ImportError as e:
        logger.error(f"TensorFlow not installed: {e}")
        yield
        return

    for path, name in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (LABELS_PATH, "Labels")]:
        if not os.path.exists(path):
            logger.error(f"{name} not found: {path}")
            yield
            return

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.predict(np.zeros((1, EXPECTED_FEATURES), dtype=np.float32), verbose=0)
        logger.info("Model loaded OK")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        yield
        return

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded OK")
    except Exception as e:
        logger.error(f"Scaler load failed: {e}")
        yield
        return

    try:
        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
        logger.info(f"Labels loaded: {labels}")
    except Exception as e:
        logger.error(f"Labels load failed: {e}")
        yield
        return

    yield
    logger.info("Shutdown.")


app = FastAPI(title="DDoS Detection API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: List[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(v)}")
        return v

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_scores: dict


@app.get("/health")
def health():
    ready = model is not None and scaler is not None and len(labels) > 0
    return {
        "status": "ok" if ready else "model_failed",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "labels_loaded": len(labels) > 0,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or scaler is None or not labels:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(request.features, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0)[0]
    top_idx = int(np.argmax(probs))

    return PredictResponse(
        predicted_class=labels[top_idx],
        confidence=round(float(probs[top_idx]), 6),
        all_scores={labels[i]: round(float(probs[i]), 6) for i in range(len(labels))},
    )