import joblib
import os
from sklearn.linear_model import LogisticRegression
from src.preprocessing_data import get_preprocessor
from src.data_loader import load_data
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)



def train(
    train_data_path = "data/train.csv",
    model_path = "models/artifacts/titanic_model.joblib",
    random_state: int = 42,
):

    # Load training data
    logger.info("Loading data for training...")
    df_train = load_data(train_data_path)
    if df_train is None:
        raise FileNotFoundError(f"Could not load training data from {train_data_path}")
            
    X_train = df_train.drop(columns=["Survived"])
    y_train = df_train["Survived"]

    # Get preprocessing pipeline
    preprocessor = get_preprocessor()

    # Transform features
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Train model
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train_transformed, y_train)

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path) or "models", exist_ok=True)
    
    artifacts = {
        "preprocessor": preprocessor,
        "model": model,
    }
    joblib.dump(artifacts, model_path)
    
    
    return {
        "preprocessor": preprocessor,
        "model": model
        }
        





"""

import os
import joblib

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_data
from src.preprocessing_data import preprocess_data

from src.utils.logger import get_logger

logger = get_logger(__name__)


def train(
    train_data_path: str = "data/train.csv",
    model_path: str = "models/titanic_model.joblib",
    random_state: int = 42,
):

    Train a simple Logistic Regression model for the Titanic survival prediction.

    Returns a dict with evaluation metrics and paths to saved artifacts.

    try:

        logger.info("Loading data for training...")
        df_train = load_data(train_data_path)
        if df_train is None:
            raise FileNotFoundError(f"Could not load training data from {train_data_path}")

        logger.info("Preprocessing data...")
        X_train, y_train = preprocess_data(df_train, is_train=True)
        

        logger.info("Training Logistic Regression model...")
        model = LogisticRegression(random_state=random_state)
        model.fit(X_train, y_train)


        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path) or "models", exist_ok=True)
        joblib.dump(model, model_path)
        
        logger.info("Training complete.")
    except FileNotFoundError as error:
        logger.error(f"File not found error during model training: {error}")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
"""