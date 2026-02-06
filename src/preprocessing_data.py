import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    numeric_features = ["Age", "SibSp", "Parch", "Pclass"]
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ["Sex", "Embarked"]
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame, is_train=True):
    
    
    Preprocess the input DataFrame by handling missing values and encoding categorical variables.
    Returns: X, y (if train) or X only (if test)
    
    
    logger.info("Starting data preprocessing...")
    df = df.copy()
    
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Number of missing values before preprocessing:{df.isnull().sum()}")
    logger.info(f"Data types before preprocessing:{df.dtypes}")

    # Fill missing values
    df.fillna({"Age": df["Age"].median()}, inplace=True) # fill with median age
    logger.info(f"Preprosess the Age feature, number of Age null value = {df['Age'].isnull().sum()} ")
    
    df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True) #fil with most frequent value
    logger.info(f"Preprosess the Embarked feature, number of Embarked null value = {df['Embarked'].isnull().sum()} ")
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True) # one hot encoding
    
    logger.info("One hot encoding for Embarked feature completed. see an example below: ")
    logger.info(f"\n{df[['Embarked_Q', 'Embarked_S']].head()}")

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    logger.info(f"Encoding for Sex feature completed. see an example:\n{df['Sex'].head()}")

    # Features to use
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
    X = df[feature_cols]
    logger.info(f"Selected features: {feature_cols}")

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Feature scaling completed.")

    if is_train:
        y = df["Survived"]
        return X_scaled, y
    else:
        return X_scaled
"""