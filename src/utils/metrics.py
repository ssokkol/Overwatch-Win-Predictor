"""Custom ML metrics for model evaluation."""

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], y_proba: NDArray[np.float_] | None = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        # Calculate binary classification metrics for probability threshold
        threshold = 0.5
        y_pred_thresh = (y_proba >= threshold).astype(int)
        metrics["accuracy_threshold"] = float(
            accuracy_score(y_true, y_pred_thresh)
        )

    return metrics


def get_classification_report(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_]
) -> str:
    """
    Get detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, zero_division=0)


def calculate_calibration_error(
    y_true: NDArray[np.int_], y_proba: NDArray[np.float_], n_bins: int = 10
) -> float:
    """
    Calculate calibration error (ECE - Expected Calibration Error).

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Calibration error score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

