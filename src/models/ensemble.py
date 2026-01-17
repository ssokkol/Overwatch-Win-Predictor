"""Ensemble model combining XGBoost and Neural Network."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.models.neural_net import NeuralNetModel
from src.models.xgboost_model import XGBoostModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EnsembleModel:
    """Ensemble model combining multiple base models."""

    def __init__(
        self,
        xgboost_model: XGBoostModel | None = None,
        neural_net_model: NeuralNetModel | None = None,
        xgboost_weight: float = 0.6,
        neural_net_weight: float = 0.4,
    ) -> None:
        """
        Initialize ensemble model.

        Args:
            xgboost_model: Trained XGBoost model
            neural_net_model: Trained Neural Network model
            xgboost_weight: Weight for XGBoost predictions
            neural_net_weight: Weight for Neural Network predictions

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if abs(xgboost_weight + neural_net_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.xgboost_model = xgboost_model
        self.neural_net_model = neural_net_model
        self.xgboost_weight = xgboost_weight
        self.neural_net_weight = neural_net_weight
        self.is_trained = (
            xgboost_model is not None
            and neural_net_model is not None
            and xgboost_model.is_trained
            and neural_net_model.is_trained
        )

    def predict(self, X: NDArray[np.float_]) -> NDArray[np.int_]:
        """
        Predict match outcomes using ensemble.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)

        Raises:
            ValueError: If models not trained
        """
        if not self.is_trained:
            raise ValueError("Ensemble models must be trained before prediction")

        proba = self.predict_proba(X)
        # Return class 1 probabilities thresholded at 0.5
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Predict match outcome probabilities using ensemble.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted probabilities (n_samples, 2) for classes [0, 1]

        Raises:
            ValueError: If models not trained
        """
        if not self.is_trained:
            raise ValueError("Ensemble models must be trained before prediction")

        # Get predictions from both models
        xgb_proba = self.xgboost_model.predict_proba(X)
        nn_proba = self.neural_net_model.predict_proba(X)

        # Weighted average
        ensemble_proba = (
            self.xgboost_weight * xgb_proba + self.neural_net_weight * nn_proba
        )

        return ensemble_proba

    def add_xgboost_model(self, model: XGBoostModel) -> None:
        """
        Add XGBoost model to ensemble.

        Args:
            model: Trained XGBoost model
        """
        self.xgboost_model = model
        self.is_trained = (
            self.xgboost_model is not None
            and self.neural_net_model is not None
            and self.xgboost_model.is_trained
            and self.neural_net_model.is_trained
        )

    def add_neural_net_model(self, model: NeuralNetModel) -> None:
        """
        Add Neural Network model to ensemble.

        Args:
            model: Trained Neural Network model
        """
        self.neural_net_model = model
        self.is_trained = (
            self.xgboost_model is not None
            and self.neural_net_model is not None
            and self.xgboost_model.is_trained
            and self.neural_net_model.is_trained
        )

    def save(self, model_dir: Path) -> None:
        """
        Save ensemble model to directory.

        Args:
            model_dir: Directory to save models
        """
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.xgboost_model:
            self.xgboost_model.save(model_dir / "xgboost_model.pkl")

        if self.neural_net_model:
            self.neural_net_model.save(model_dir / "neural_net_model.pt")

        # Save ensemble metadata
        import json

        metadata = {
            "xgboost_weight": self.xgboost_weight,
            "neural_net_weight": self.neural_net_weight,
        }
        with open(model_dir / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f)

        logger.info(f"Ensemble model saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: Path, device: str = "cpu") -> "EnsembleModel":
        """
        Load ensemble model from directory.

        Args:
            model_dir: Directory containing model files
            device: Device for neural network model

        Returns:
            Loaded EnsembleModel instance

        Raises:
            FileNotFoundError: If model files not found
        """
        xgb_path = model_dir / "xgboost_model.pkl"
        nn_path = model_dir / "neural_net_model.pt"
        metadata_path = model_dir / "ensemble_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")

        # Load metadata
        import json

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load models
        xgboost_model = None
        if xgb_path.exists():
            xgboost_model = XGBoostModel.load(xgb_path)

        neural_net_model = None
        if nn_path.exists():
            neural_net_model = NeuralNetModel.load(nn_path, device=device)

        instance = cls(
            xgboost_model=xgboost_model,
            neural_net_model=neural_net_model,
            xgboost_weight=metadata["xgboost_weight"],
            neural_net_weight=metadata["neural_net_weight"],
        )

        logger.info(f"Ensemble model loaded from {model_dir}")

        return instance
