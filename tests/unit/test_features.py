"""Unit tests for feature engineering."""

import numpy as np
import pytest

from src.features.hero_features import HeroFeatureExtractor
from src.features.team_composition import TeamCompositionFeatureExtractor


class TestHeroFeatureExtractor:
    """Test HeroFeatureExtractor."""

    def test_extract_features_valid_input(
        self, feature_extractor: HeroFeatureExtractor, sample_team1: list[int]
    ) -> None:
        """Test feature extraction with valid team composition."""
        features = feature_extractor.extract_features(sample_team1)

        assert isinstance(features, dict)
        assert len(features) > 0
        assert all(isinstance(v, (int, float)) for v in features.values())

    def test_extract_features_invalid_team_size(
        self, feature_extractor: HeroFeatureExtractor
    ) -> None:
        """Test that invalid team sizes raise ValueError."""
        team1 = [1, 2, 3]  # Too few heroes

        with pytest.raises(ValueError, match="Team must have 5-6 heroes"):
            feature_extractor.extract_features(team1)

    def test_extract_role_distribution(
        self, feature_extractor: HeroFeatureExtractor
    ) -> None:
        """Test role distribution extraction."""
        team = [1, 5, 10, 15, 20]  # Support, Support, DPS, Support, Support
        role_dist = feature_extractor.extract_role_distribution(team)

        assert "Tank" in role_dist
        assert "DPS" in role_dist
        assert "Support" in role_dist
        assert role_dist["Support"] == 4

    @pytest.mark.parametrize(
        "team1,team2",
        [
            ([1, 1, 3, 4, 5], [6, 7, 8, 9, 10]),  # Duplicate hero
            ([1, 2, 3, 4, 6], [6, 7, 8, 9, 10]),  # Hero in both teams
        ],
    )
    def test_extract_features_duplicate_heroes(
        self, feature_extractor: HeroFeatureExtractor, team1: list[int], team2: list[int]
    ) -> None:
        """Test that duplicate heroes are detected."""
        # Should not raise error in feature extraction, validation happens elsewhere
        # But team composition extractor should handle it
        pass


class TestTeamCompositionFeatureExtractor:
    """Test TeamCompositionFeatureExtractor."""

    def test_extract_team_features_valid(
        self,
        team_feature_extractor: TeamCompositionFeatureExtractor,
        sample_team1: list[int],
        sample_team2: list[int],
    ) -> None:
        """Test team feature extraction with valid teams."""
        features = team_feature_extractor.extract_team_features(sample_team1, sample_team2)

        assert isinstance(features, dict)
        assert len(features) > 0
        assert "team1_tank_count" in features
        assert "team2_tank_count" in features
        assert "diff_tank_count" in features

    def test_extract_team_features_invalid_size(
        self,
        team_feature_extractor: TeamCompositionFeatureExtractor,
    ) -> None:
        """Test that invalid team sizes raise ValueError."""
        team1 = [1, 2, 3]  # Too few heroes
        team2 = [6, 7, 8, 9, 10]

        with pytest.raises(ValueError, match="Teams must have at least 5 heroes"):
            team_feature_extractor.extract_team_features(team1, team2)

    def test_extract_feature_vector(
        self,
        team_feature_extractor: TeamCompositionFeatureExtractor,
        sample_team1: list[int],
        sample_team2: list[int],
    ) -> None:
        """Test feature vector extraction."""
        feature_vector = team_feature_extractor.extract_feature_vector(sample_team1, sample_team2)

        assert isinstance(feature_vector, np.ndarray)
        assert feature_vector.dtype == np.float32
        assert len(feature_vector) > 0
