import joblib
import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

artifacts = joblib.load("models/artifacts/titanic_model.joblib")
preprocessor = artifacts["preprocessor"]
model = artifacts["model"]

def predict(df: pd.DataFrame):
    logger.info(f"Received {len(df)} rows for prediction")
    X_transformed = preprocessor.transform(df)
    predictions = model.predict(X_transformed)
    return predictions


"""
import os
import joblib

from src.data_loader import load_data
from src.preprocessing_data import preprocess_data

from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict(
    data_path: str = "data/test.csv",
    model_path: str = "models/titanic_model.joblib"
):
    
    Load a saved model, preprocess input data, produce predictions.

    Returns a dict with predictions DataFrame (under 'predictions').
    
    try:
        logger.info(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)

        logger.info(f"Loading testing data from {data_path}...")
        df = load_data(data_path)
        if df is None:
            raise FileNotFoundError(f"Could not load data from {data_path}")

        logger.info("Preprocessing input data...")
        X = preprocess_data(df, is_train=False)

        # Log shape/type for debugging
        try:
            logger.info(f"Preprocessed features type: {type(X)}, shape: {getattr(X, 'shape', None)}")
        except Exception:
            logger.info(f"Preprocessed features type: {type(X)}")

        logger.info("Running model predictions...")
        y_pred = model.predict(X)
        # Log prediction type/shape for debugging
        try:
            logger.info(f"Predictions type: {type(y_pred)}, shape: {getattr(y_pred, 'shape', None)}")
        except Exception:
            logger.info(f"Predictions type: {type(y_pred)}")
        logger.info("Predictions completed.")
    except FileNotFoundError as error:
        logger.error(f"File not found error during prediction: {error}")
        return None
    return  y_pred
"""