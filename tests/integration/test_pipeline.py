"""Integration tests for data pipeline."""

import numpy as np

from src.data.generate_synthetic import generate_synthetic_data
from src.data.preprocess import load_match_data
from src.features.team_composition import TeamCompositionFeatureExtractor


class TestDataPipeline:
    """Test data pipeline integration."""

    def test_synthetic_data_generation(self, tmp_path: any) -> None:
        """Test synthetic data generation."""
        output_path = tmp_path / "test_matches.csv"
        df = generate_synthetic_data(n_matches=100, output_path=output_path)

        assert len(df) == 100
        assert "winner" in df.columns
        assert all(col.startswith("team") for col in df.columns if "hero" in col)

    def test_data_loading(self, tmp_path: any) -> None:
        """Test data loading."""
        output_path = tmp_path / "test_matches.csv"
        generate_synthetic_data(n_matches=50, output_path=output_path)

        df = load_match_data(output_path)
        assert len(df) == 50

    def test_feature_extraction_pipeline(
        self, team_feature_extractor: TeamCompositionFeatureExtractor
    ) -> None:
        """Test feature extraction pipeline."""
        team1 = [1, 5, 10, 15, 20]
        team2 = [2, 7, 12, 17, 22]

        features = team_feature_extractor.extract_feature_vector(team1, team2)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
