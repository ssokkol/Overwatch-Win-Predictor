"""XGBoost model for match prediction."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class XGBoostModel:
    """XGBoost classifier for match prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        early_stopping_rounds: int = 10,
    ) -> None:
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.is_trained = False

    def train(
        self,
        X_train: NDArray[np.float_],
        y_train: NDArray[np.int_],
        X_val: NDArray[np.float_] | None = None,
        y_val: NDArray[np.int_] | None = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Tuple of (training_accuracy, metrics_dict)
        """
        logger.info(f"Training XGBoost model on {len(X_train)} samples")

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            verbose=False,
        )

        # Evaluate on training data
        y_pred = self.model.predict(X_train)
        y_proba = self.model.predict_proba(X_train)[:, 1]

        train_accuracy = float(np.mean(y_pred == y_train))

        from src.utils.metrics import calculate_metrics

        metrics = calculate_metrics(y_train, y_pred, y_proba)
        self.is_trained = True

        logger.info(f"XGBoost model training complete. Accuracy: {train_accuracy:.4f}")

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

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        importance = self.model.feature_importances_
        # Return as dict with feature indices as keys
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def save(self, model_path: Path) -> None:
        """
        Save model to disk.

        Args:
            model_path: Path to save model file
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        logger.info(f"XGBoost model saved to {model_path}")

    @classmethod
    def load(cls, model_path: Path) -> "XGBoostModel":
        """
        Load model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Loaded XGBoostModel instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        instance = cls()
        instance.model = joblib.load(model_path)
        instance.is_trained = True
        logger.info(f"XGBoost model loaded from {model_path}")

        return instance

