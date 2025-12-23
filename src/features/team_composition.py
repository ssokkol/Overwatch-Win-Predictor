"""Extract team-level composition features."""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from src.features.hero_features import HeroFeatureExtractor
from src.utils.heroes import get_hero_metadata


class TeamCompositionFeatureExtractor:
    """Extract features comparing two team compositions."""

    def __init__(self, embeddings: NDArray[np.float32] | None = None) -> None:
        """
        Initialize team composition feature extractor.

        Args:
            embeddings: Hero embedding matrix
        """
        self.metadata = get_hero_metadata()
        self.hero_extractor = HeroFeatureExtractor(embeddings)
        self.embeddings = embeddings

    def extract_team_features(
        self, team1: List[int], team2: List[int]
    ) -> Dict[str, float]:
        """
        Extract features comparing two teams.

        Args:
            team1: First team hero IDs
            team2: Second team hero IDs

        Returns:
            Dictionary of team comparison features

        Raises:
            ValueError: If team sizes are invalid
        """
        if len(team1) < 5 or len(team2) < 5:
            raise ValueError("Teams must have at least 5 heroes")

        features: Dict[str, float] = {}

        # Individual team features
        team1_features = self.hero_extractor.extract_features(team1)
        team2_features = self.hero_extractor.extract_features(team2)

        # Prefix team features
        for key, value in team1_features.items():
            features[f"team1_{key}"] = value
        for key, value in team2_features.items():
            features[f"team2_{key}"] = value

        # Difference features
        for key in team1_features:
            if key in team2_features:
                features[f"diff_{key}"] = team1_features[key] - team2_features[key]

        # Embedding-based team similarity
        if self.embeddings is not None:
            embedding_features = self._extract_embedding_comparison(team1, team2)
            features.update(embedding_features)

        # Synergy and counter features
        synergy_features = self._extract_synergy_features(team1, team2)
        features.update(synergy_features)

        return features

    def _extract_embedding_comparison(
        self, team1: List[int], team2: List[int]
    ) -> Dict[str, float]:
        """
        Extract embedding-based comparison features.

        Args:
            team1: First team hero IDs
            team2: Second team hero IDs

        Returns:
            Dictionary of embedding comparison features
        """
        if self.embeddings is None:
            return {}

        # Get team embeddings
        team1_embeddings = np.array(
            [
                self.embeddings[hid - 1]
                for hid in team1
                if 1 <= hid <= len(self.embeddings)
            ]
        )
        team2_embeddings = np.array(
            [
                self.embeddings[hid - 1]
                for hid in team2
                if 1 <= hid <= len(self.embeddings)
            ]
        )

        if len(team1_embeddings) == 0 or len(team2_embeddings) == 0:
            return {}

        # Team centroids
        team1_centroid = np.mean(team1_embeddings, axis=0)
        team2_centroid = np.mean(team2_embeddings, axis=0)

        # Within-team similarity
        team1_similarities = []
        for i in range(len(team1_embeddings)):
            for j in range(i + 1, len(team1_embeddings)):
                sim = np.dot(team1_embeddings[i], team1_embeddings[j])
                team1_similarities.append(sim)

        team2_similarities = []
        for i in range(len(team2_embeddings)):
            for j in range(i + 1, len(team2_embeddings)):
                sim = np.dot(team2_embeddings[i], team2_embeddings[j])
                team2_similarities.append(sim)

        # Between-team distance
        centroid_distance = np.linalg.norm(team1_centroid - team2_centroid)

        return {
            "team1_internal_similarity": float(np.mean(team1_similarities)) if team1_similarities else 0.0,
            "team2_internal_similarity": float(np.mean(team2_similarities)) if team2_similarities else 0.0,
            "centroid_distance": float(centroid_distance),
            "similarity_diff": float(np.mean(team1_similarities) - np.mean(team2_similarities)) if (team1_similarities and team2_similarities) else 0.0,
        }

    def _extract_synergy_features(
        self, team1: List[int], team2: List[int]
    ) -> Dict[str, float]:
        """
        Extract synergy and counter features.

        Args:
            team1: First team hero IDs
            team2: Second team hero IDs

        Returns:
            Dictionary of synergy features
        """
        # Synergy pairs (heroes that work well together)
        synergy_pairs = [
            (1, 25),   # Ana + Reinhardt
            (7, 16),   # D.Va + Lucio
            (14, 28),  # Kiriko + Sojourn
            (20, 25),  # Moira + Reinhardt
            (35, 16),  # Winston + Lucio
        ]

        team1_set = set(team1)
        team2_set = set(team2)

        # Count synergies in each team
        team1_synergies = sum(
            1
            for h1, h2 in synergy_pairs
            if h1 in team1_set and h2 in team1_set
        )
        team2_synergies = sum(
            1
            for h1, h2 in synergy_pairs
            if h1 in team2_set and h2 in team2_set
        )

        return {
            "team1_synergy_count": float(team1_synergies),
            "team2_synergy_count": float(team2_synergies),
            "synergy_diff": float(team1_synergies - team2_synergies),
        }

    def extract_feature_vector(
        self, team1: List[int], team2: List[int]
    ) -> NDArray[np.float32]:
        """
        Extract feature vector (for ML models).

        Args:
            team1: First team hero IDs
            team2: Second team hero IDs

        Returns:
            Feature vector as numpy array
        """
        features = self.extract_team_features(team1, team2)
        # Sort by key for consistent ordering
        feature_values = [features[key] for key in sorted(features.keys())]
        return np.array(feature_values, dtype=np.float32)

