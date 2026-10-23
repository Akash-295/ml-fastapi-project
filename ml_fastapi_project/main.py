
import math
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the trusted model artifact once at startup
artifact = joblib.load("model_artifact.joblib")

model = artifact["model"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

app = FastAPI(
    title="ML Prediction API",
    description="A machine learning prediction service",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        description="Feature values in the documented order"
    )


@app.get("/")
def home():
    return {
        "message": "ML Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    values = request.features

    # Validate the feature count
    if len(values) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(feature_names)} features, "
                   f"received {len(values)}"
        )

    # Reject NaN and infinite values
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(
            status_code=422,
            detail="All features must be finite numbers"
        )

    # Preserve feature names and their order
    input_df = pd.DataFrame(
        [values],
        columns=feature_names
    )

    # Predict with the complete fitted pipeline
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": prediction,
        "predicted_class": target_names[prediction],
        "probabilities": {
            target_names[int(class_id)]: float(probability)
            for class_id, probability
            in zip(model.classes_, probabilities)
        }
    }
