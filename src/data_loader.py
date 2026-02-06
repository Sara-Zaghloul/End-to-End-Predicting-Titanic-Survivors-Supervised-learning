import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)



def load_data(path: str):
    """
    Load a CSV file and return a pandas DataFrame.
    """
    try:
        data= pd.read_csv(path)
        logger.info(f"Data loaded successfully with shape {data.shape}")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")