"""Shared pytest fixtures."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.features.hero_features import HeroFeatureExtractor
from src.features.team_composition import TeamCompositionFeatureExtractor
from src.models.baseline import BaselineModel
from src.models.ensemble import EnsembleModel
from src.models.neural_net import NeuralNetModel
from src.models.xgboost_model import XGBoostModel
from src.utils.heroes import get_hero_metadata


@pytest.fixture
def sample_team1() -> list[int]:
    """Sample team 1 composition."""
    return [1, 5, 10, 15, 20]  # Ana, Brigitte, Genji, Lifeweaver, Mercy


@pytest.fixture
def sample_team2() -> list[int]:
    """Sample team 2 composition."""
    return [2, 7, 12, 17, 21]  # Ashe, D.Va, Junker Queen, Mauga, Orisa


@pytest.fixture(autouse=True)
def _allow_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow placeholder model during tests when artifacts are missing."""
    monkeypatch.setenv("ALLOW_MISSING_MODEL", "true")


@pytest.fixture
def hero_metadata():
    """Hero metadata fixture."""
    return get_hero_metadata()


@pytest.fixture
def embeddings() -> NDArray[np.float32]:
    """Sample hero embeddings."""
    metadata = get_hero_metadata()
    n_heroes = len(metadata.heroes)
    return np.random.rand(n_heroes, 32).astype(np.float32)


@pytest.fixture
def feature_extractor(embeddings: NDArray[np.float32]) -> HeroFeatureExtractor:
    """Feature extractor fixture."""
    return HeroFeatureExtractor(embeddings=embeddings)


@pytest.fixture
def team_feature_extractor(embeddings: NDArray[np.float32]) -> TeamCompositionFeatureExtractor:
    """Team composition feature extractor fixture."""
    return TeamCompositionFeatureExtractor(embeddings=embeddings)


@pytest.fixture
def sample_features(team_feature_extractor: TeamCompositionFeatureExtractor, sample_team1: list[int], sample_team2: list[int]) -> NDArray[np.float32]:
    """Sample feature vector."""
    return team_feature_extractor.extract_feature_vector(sample_team1, sample_team2)


@pytest.fixture
def sample_training_data(
    sample_features: NDArray[np.float32],
) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
    """Sample training data."""
    n_samples = 100
    n_features = sample_features.shape[0]
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.int_)
    return X, y


@pytest.fixture
def baseline_model() -> BaselineModel:
    """Baseline model fixture."""
    return BaselineModel(random_state=42)


@pytest.fixture
def xgboost_model() -> XGBoostModel:
    """XGBoost model fixture."""
    return XGBoostModel(random_state=42, n_estimators=10)


@pytest.fixture
def neural_net_model() -> NeuralNetModel:
    """Neural network model fixture."""
    return NeuralNetModel(input_dim=50, epochs=5, random_state=42)


@pytest.fixture
def trained_baseline_model(baseline_model: BaselineModel, sample_training_data: tuple) -> BaselineModel:
    """Trained baseline model fixture."""
    X, y = sample_training_data
    baseline_model.train(X, y)
    return baseline_model


@pytest.fixture
def trained_xgboost_model(xgboost_model: XGBoostModel, sample_training_data: tuple) -> XGBoostModel:
    """Trained XGBoost model fixture."""
    X, y = sample_training_data
    xgboost_model.train(X, y)
    return xgboost_model


@pytest.fixture
def trained_neural_net_model(neural_net_model: NeuralNetModel, sample_training_data: tuple) -> NeuralNetModel:
    """Trained neural network model fixture."""
    X, y = sample_training_data
    neural_net_model.train(X, y)
    return neural_net_model


@pytest.fixture
def ensemble_model(trained_xgboost_model: XGBoostModel, trained_neural_net_model: NeuralNetModel) -> EnsembleModel:
    """Ensemble model fixture."""
    ensemble = EnsembleModel(
        xgboost_model=trained_xgboost_model,
        neural_net_model=trained_neural_net_model,
        xgboost_weight=0.6,
        neural_net_weight=0.4,
    )
    return ensemble

