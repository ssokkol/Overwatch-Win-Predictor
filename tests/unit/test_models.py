"""Unit tests for ML models."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.models.baseline import BaselineModel
from src.models.ensemble import EnsembleModel
from src.models.neural_net import NeuralNetModel
from src.models.xgboost_model import XGBoostModel


class TestBaselineModel:
    """Test BaselineModel."""

    def test_train_and_predict(
        self, baseline_model: BaselineModel, sample_training_data: tuple
    ) -> None:
        """Test model training and prediction."""
        X, y = sample_training_data

        accuracy, metrics = baseline_model.train(X, y)

        assert 0.0 <= accuracy <= 1.0
        assert "accuracy" in metrics
        assert baseline_model.is_trained

        predictions = baseline_model.predict(X)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

        proba = baseline_model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_before_training(self, baseline_model: BaselineModel) -> None:
        """Test that prediction fails before training."""
        X = np.random.rand(10, 5)

        with pytest.raises(ValueError, match="Model must be trained"):
            baseline_model.predict(X)


class TestXGBoostModel:
    """Test XGBoostModel."""

    def test_train_and_predict(
        self, xgboost_model: XGBoostModel, sample_training_data: tuple
    ) -> None:
        """Test XGBoost training and prediction."""
        X, y = sample_training_data

        accuracy, metrics = xgboost_model.train(X, y)

        assert 0.0 <= accuracy <= 1.0
        assert xgboost_model.is_trained

        predictions = xgboost_model.predict(X)
        assert len(predictions) == len(X)

        proba = xgboost_model.predict_proba(X)
        assert proba.shape == (len(X), 2)

        importance = xgboost_model.get_feature_importance()
        assert isinstance(importance, dict)


class TestNeuralNetModel:
    """Test NeuralNetModel."""

    def test_train_and_predict(
        self, neural_net_model: NeuralNetModel, sample_training_data: tuple
    ) -> None:
        """Test neural network training and prediction."""
        X, y = sample_training_data

        accuracy, metrics = neural_net_model.train(X, y)

        assert 0.0 <= accuracy <= 1.0
        assert neural_net_model.is_trained

        predictions = neural_net_model.predict(X)
        assert len(predictions) == len(X)

        proba = neural_net_model.predict_proba(X)
        assert proba.shape == (len(X), 2)


class TestEnsembleModel:
    """Test EnsembleModel."""

    def test_ensemble_prediction(
        self,
        ensemble_model: EnsembleModel,
        sample_features: NDArray[np.float32],
    ) -> None:
        """Test ensemble model prediction."""
        X = sample_features.reshape(1, -1)

        predictions = ensemble_model.predict(X)
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]

        proba = ensemble_model.predict_proba(X)
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_ensemble_invalid_weights(self) -> None:
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            EnsembleModel(xgboost_weight=0.6, neural_net_weight=0.5)

