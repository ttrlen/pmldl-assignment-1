import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/exoplanet_classifier.joblib"))
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    koi_period: float = Field(gt=0)
    koi_duration: float = Field(gt=0)
    koi_depth: float = Field(ge=0)
    koi_prad: float = Field(gt=0)
    koi_teq: float = Field(gt=0)
    koi_insol: float = Field(ge=0)
    koi_model_snr: float = Field(ge=0)
    koi_steff: float = Field(gt=0)
    koi_slogg: float = Field(gt=0)
    koi_srad: float = Field(gt=0)
    koi_kepmag: float = Field(gt=0)


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


app = FastAPI(title="Exoplanet Classification API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    dataframe = pd.DataFrame([request.model_dump()])
    prediction = str(model.predict(dataframe)[0])
    probabilities = model.predict_proba(dataframe)[0]
    probability_by_class = {
        str(label): float(probability)
        for label, probability in zip(model.named_steps["classifier"].classes_, probabilities, strict=True)
    }
    return PredictionResponse(prediction=prediction, probabilities=probability_by_class)
