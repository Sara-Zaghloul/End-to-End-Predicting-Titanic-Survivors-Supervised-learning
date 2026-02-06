import os
import pandas as pd
import joblib

from src.data_loader import load_data
from src.preprocessing_data import get_preprocessor
from src.train_model import train
from src.predict import predict
from src.evaluate import evaluate

from routes.ml_routes import Passenger, PredictionResponse

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)

"""
try:
    res = train()
    logger.debug(f"Train results: {res}")
    
    # Load test data for predictions
    df_test = load_data("data/test.csv")
    y_pred = predict(df_test)
    logger.debug(f"Predictions: {y_pred}")
    
    # keep a copy of true labels if present
    df_val = load_data("data/gender_submission.csv")
    logger.debug(f"Validation data loaded: {df_val.shape}")
    
    y_true = df_val["Survived"].copy() if "Survived" in df_val.columns else None

    logger.debug(f"True labels: {y_true}")

    metrics = evaluate(y_true, y_pred)
    logger.debug(f"Evaluation metrics: {metrics}")
    logger.info("Pipeline execution completed.")
    
except Exception as e:
    logger.error(f"An error occurred during pipeline execution: {e}")

# Load model once at startup
artifacts = joblib.load("models/artifacts/titanic_model.joblib")
preprocessor = artifacts["preprocessor"]
model = artifacts["model"]
"""

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API de prédiction ML avec FastAPI",
    version="1.0"
)



@app.get("/")
def health_check():
    return {"status": "Titanic Survival Prediction Pipeline is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_survival(passenger: Passenger):
    # Log incoming request
    logger.info(f"Single prediction requested: {passenger.dict()}")
    data = pd.DataFrame([passenger.dict()])
    
    # Preprocess the data before prediction
    data_transformed = preprocessor.transform(data)
    prediction = model.predict(data_transformed)[0]
    
    return PredictionResponse(
        survived=int(prediction),
        message="Passenger survived" if prediction == 1 else "Passenger did not survive"
    )
    



 