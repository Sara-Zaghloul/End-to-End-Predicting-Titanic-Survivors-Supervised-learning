
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(y_true, y_pred):
    """
    Compute common classification metrics and return them in a dict.

    This function normalizes inputs to 1-D numpy arrays and validates that
    `y_true` and `y_pred` have the same number of samples before computing
    metrics. Raises `ValueError` with a clear message if validation fails.
    """

    # Convert to numpy arrays and flatten
    y_pred_arr = np.asarray(y_pred).ravel()
    y_true_arr = None if y_true is None else np.asarray(y_true).ravel()

    logger.info(f"Evaluating: y_true type={type(y_true_arr)}, shape={None if y_true_arr is None else y_true_arr.shape}; y_pred type={type(y_pred_arr)}, shape={y_pred_arr.shape}")

    if y_true_arr is None:
        logger.error("y_true is None; cannot compute evaluation metrics without true labels.")
        raise ValueError("y_true is None; cannot compute evaluation metrics without true labels.")

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        logger.error(f"Inconsistent sample counts: y_true={y_true_arr.shape[0]}, y_pred={y_pred_arr.shape[0]}")
        raise ValueError(f"Inconsistent sample counts: y_true={y_true_arr.shape[0]}, y_pred={y_pred_arr.shape[0]}")

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
    report = classification_report(y_true_arr, y_pred_arr, output_dict=True)
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    logger.info(f"Metrics - acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }
