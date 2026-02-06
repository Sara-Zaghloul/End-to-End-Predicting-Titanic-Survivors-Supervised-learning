# Titanic Survival Prediction

An end-to-end machine learning project that predicts passenger survival on the Titanic using a classification model built with scikit-learn. The project includes data preprocessing, model training, evaluation, and a FastAPI-based REST API for making predictions.


## Overview

This project implements a complete machine learning pipeline for the Kaggle Titanic dataset. It:

- **Loads and preprocesses** the Titanic dataset
- **Trains a Logistic Regression** classifier
- **Evaluates** model performance on test data
- **Serves predictions** via a FastAPI REST API
- **Logs operations** with a custom logging utility

The model predicts whether a passenger survived the Titanic disaster based on features like age, gender, passenger class, and fare.

## Project Structure

```
titanic_end_to_end/
├── main.py                          # Main entry point - runs pipeline & API
├── requirements.txt                 # Project dependencies
├── README.md                        # This file
│
├── data/                            # Dataset files
│   ├── train.csv                   # Training data
│   ├── test.csv                    # Test data
│   └── gender_submission.csv       # Ground truth labels
│
├── src/                             # Source code
│   ├── __init__.py
│   ├── data_loader.py              # Data loading utilities
│   ├── preprocessing_data.py       # Feature preprocessing pipeline
│   ├── train_model.py              # Model training logic
│   ├── predict.py                  # Prediction logic
│   ├── evaluate.py                 # Model evaluation metrics
│   └── utils/
│       ├── logger.py               # Custom logging utility
│       └── exceptions.py           # Custom exceptions
│
├── routes/                          # API routes
│   ├── __init__.py
│   └── ml_routes.py                # FastAPI endpoints
│
├── models/
│   └── artifacts/
│       └── titanic_model.joblib   # Trained model artifacts
│
├── notebook/
│   └── titanic_project.ipynb      # Jupyter notebook for exploration
│
└── logs/                            # Log files
```

### Running the API

The FastAPI server starts automatically when you run `main.py`. Access it at:

```
http://localhost:8000
```

View the interactive API documentation:

```
http://localhost:8000/docs
```

## API Endpoints

### POST `/predict`

Make a survival prediction for a single passenger.

**Request Body:**
```json
{
  "PassengerId": 892,
  "Pclass": 3,
  "Name": "John Doe",
  "Sex": "male",
  "Age": 34.5,
  "SibSp": 0,
  "Parch": 0,
  "Ticket": "350029",
  "Fare": 7.75,
  "Cabin": "F38",
  "Embarked": "S"
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.25,
  "message": "Passenger did not survive"
}
```

## Project Components

### Data Loader (`src/data_loader.py`)
Handles loading CSV files and returning pandas DataFrames.

### Preprocessing (`src/preprocessing_data.py`)
Implements feature engineering and preprocessing:
- Handles missing values
- Encodes categorical variables
- Scales numerical features

### Model Training (`src/train_model.py`)
- Trains a Logistic Regression classifier
- Saves model and preprocessor artifacts
- Implements custom error handling

### Predictions (`src/predict.py`)
- Loads trained model
- Makes predictions on new data
- Handles feature preprocessing

### Evaluation (`src/evaluate.py`)
- Calculates performance metrics (accuracy, precision, recall, F1-score)
- Compares predictions against ground truth

### Logging (`src/utils/logger.py`)
- Custom logger configuration
- Logs to console and file
- Tracks model pipeline execution

### API Routes (`routes/ml_routes.py`)
- FastAPI application setup
- Defines Pydantic models for request/response validation
- Implements `/predict` endpoint

## Dataset

**Source:** [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

**Features:**
- `PassengerId`: Unique identifier
- `Pclass`: Passenger class (1, 2, 3)
- `Name`: Passenger name
- `Sex`: Gender (male, female)
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Ticket fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C, Q, S)

**Target:** `Survived` (0 = No, 1 = Yes)

