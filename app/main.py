# app/main.py
# FastAPI application serving predictions from a trained heart disease model.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import HeartInput, PredictionOutput
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI + scikit-learn demo (Docker + Render deploy)",
    version="1.0.0",
)

# Allow local dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/heart_model.joblib")
_model_bundle = joblib.load(MODEL_PATH)
model = _model_bundle["model"]
FEATURES = _model_bundle["features"]

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: HeartInput):
    # Convert to a single-row DataFrame in the same column order the model expects
    df = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)
    proba = float(model.predict_proba(df)[0][1])  # positive class probability
    pred_bool = bool(proba >= 0.5)
    return {"heart_disease": pred_bool, "probability": round(proba, 4)}
