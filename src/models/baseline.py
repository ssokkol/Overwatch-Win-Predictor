"""Baseline model for match prediction."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaselineModel:
    """Simple logistic regression baseline model."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000) -> None:
        """
        Initialize baseline model.

        Args:
            random_state: Random seed
            max_iter: Maximum iterations for training
        """
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",
        )
        self.random_state = random_state
        self.max_iter = max_iter
        self.is_trained = False

    def train(
        self, X: NDArray[np.float_], y: NDArray[np.int_]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train the baseline model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Tuple of (training_accuracy, metrics_dict)
        """
        logger.info(f"Training baseline model on {len(X)} samples")

        self.model.fit(X, y)

        # Evaluate on training data
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        train_accuracy = float(np.mean(y_pred == y))

        from src.utils.metrics import calculate_metrics

        metrics = calculate_metrics(y, y_pred, y_proba)
        self.is_trained = True

        logger.info(f"Baseline model training complete. Accuracy: {train_accuracy:.4f}")

        return train_accuracy, metrics

    def predict(self, X: NDArray[np.float_]) -> NDArray[np.int_]:
        """
        Predict match outcomes.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)

        Raises:
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Predict match outcome probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted probabilities (n_samples, 2) for classes [0, 1]

        Raises:
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict_proba(X)

    def save(self, model_path: Path) -> None:
        """
        Save model to disk.

        Args:
            model_path: Path to save model file
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        logger.info(f"Baseline model saved to {model_path}")

    @classmethod
    def load(cls, model_path: Path) -> "BaselineModel":
        """
        Load model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Loaded BaselineModel instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        instance = cls()
        instance.model = joblib.load(model_path)
        instance.is_trained = True
        logger.info(f"Baseline model loaded from {model_path}")

        return instance

