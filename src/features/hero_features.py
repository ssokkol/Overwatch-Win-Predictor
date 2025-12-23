"""Extract hero-level features from team compositions."""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from src.utils.heroes import get_hero_metadata


class HeroFeatureExtractor:
    """Extract features from hero selections."""

    def __init__(self, embeddings: NDArray[np.float32] | None = None) -> None:
        """
        Initialize feature extractor.

        Args:
            embeddings: Hero embedding matrix (n_heroes, embedding_dim)
        """
        self.metadata = get_hero_metadata()
        self.embeddings = embeddings

        # Calculate hero statistics (pick rates, win rates) from training data
        # For now, use uniform distribution as placeholder
        self.hero_pick_rates: Dict[int, float] = {
            hero_id: 1.0 / len(self.metadata.heroes) for hero_id in self.metadata.heroes.keys()
        }
        self.hero_win_rates: Dict[int, float] = {
            hero_id: 0.5 for hero_id in self.metadata.heroes.keys()
        }

    def extract_role_distribution(self, hero_ids: List[int]) -> Dict[str, int]:
        """
        Extract role distribution features.

        Args:
            hero_ids: List of hero IDs

        Returns:
            Dictionary with role counts
        """
        return self.metadata.get_role_distribution(hero_ids)

    def extract_statistical_features(
        self, hero_ids: List[int]
    ) -> Dict[str, float]:
        """
        Extract statistical features (pick rates, win rates).

        Args:
            hero_ids: List of hero IDs

        Returns:
            Dictionary of statistical features
        """
        pick_rates = [self.hero_pick_rates.get(hid, 0.0) for hid in hero_ids]
        win_rates = [self.hero_win_rates.get(hid, 0.5) for hid in hero_ids]

        return {
            "avg_pick_rate": float(np.mean(pick_rates)),
            "max_pick_rate": float(np.max(pick_rates)),
            "min_pick_rate": float(np.min(pick_rates)),
            "avg_win_rate": float(np.mean(win_rates)),
            "max_win_rate": float(np.max(win_rates)),
            "min_win_rate": float(np.min(win_rates)),
        }

    def extract_embedding_features(
        self, hero_ids: List[int]
    ) -> Dict[str, float]:
        """
        Extract features from hero embeddings.

        Args:
            hero_ids: List of hero IDs

        Returns:
            Dictionary of embedding-based features
        """
        if self.embeddings is None:
            return {}

        # Get embeddings for selected heroes (convert 1-indexed to 0-indexed)
        hero_embeddings = np.array(
            [self.embeddings[hid - 1] for hid in hero_ids if 1 <= hid <= len(self.embeddings)]
        )

        if len(hero_embeddings) == 0:
            return {}

        # Calculate statistics
        mean_embedding = np.mean(hero_embeddings, axis=0)
        std_embedding = np.std(hero_embeddings, axis=0)

        # Calculate pairwise similarity
        similarities = []
        for i in range(len(hero_embeddings)):
            for j in range(i + 1, len(hero_embeddings)):
                sim = np.dot(hero_embeddings[i], hero_embeddings[j])
                similarities.append(sim)

        features: Dict[str, float] = {
            "embedding_mean_norm": float(np.linalg.norm(mean_embedding)),
            "embedding_std_mean": float(np.mean(std_embedding)),
            "embedding_similarity_mean": float(np.mean(similarities)) if similarities else 0.0,
            "embedding_similarity_std": float(np.std(similarities)) if similarities else 0.0,
        }

        return features

    def extract_features(self, hero_ids: List[int]) -> Dict[str, float]:
        """
        Extract all features for a team composition.

        Args:
            hero_ids: List of hero IDs (team composition)

        Returns:
            Dictionary of all extracted features

        Raises:
            ValueError: If team size is invalid
        """
        if not (5 <= len(hero_ids) <= 6):
            raise ValueError(f"Team must have 5-6 heroes, got {len(hero_ids)}")

        features: Dict[str, float] = {}

        # Role distribution
        role_dist = self.extract_role_distribution(hero_ids)
        features["tank_count"] = float(role_dist["Tank"])
        features["dps_count"] = float(role_dist["DPS"])
        features["support_count"] = float(role_dist["Support"])

        # Role balance score (optimal: 1-2-2)
        optimal_tank = 1.0
        optimal_dps = 2.0
        optimal_support = 2.0
        features["role_balance_score"] = float(
            1.0
            - (
                abs(features["tank_count"] - optimal_tank)
                + abs(features["dps_count"] - optimal_dps)
                + abs(features["support_count"] - optimal_support)
            )
            / 4.0
        )

        # Statistical features
        stats = self.extract_statistical_features(hero_ids)
        features.update(stats)

        # Embedding features
        embedding_features = self.extract_embedding_features(hero_ids)
        features.update(embedding_features)

        return features

    def update_statistics(
        self,
        hero_pick_rates: Dict[int, float],
        hero_win_rates: Dict[int, float],
    ) -> None:
        """
        Update hero statistics from training data.

        Args:
            hero_pick_rates: Dictionary mapping hero IDs to pick rates
            hero_win_rates: Dictionary mapping hero IDs to win rates
        """
        self.hero_pick_rates = hero_pick_rates
        self.hero_win_rates = hero_win_rates

