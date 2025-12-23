"""Generate synthetic training data for Overwatch match predictions."""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.utils.heroes import get_hero_metadata

# Meta heroes with higher win rates (example heuristics)
META_HEROES = [1, 7, 14, 20, 25, 28, 35]  # Ana, D.Va, Kiriko, Moira, Reinhardt, Sojourn, Winston

# Synergy pairs (heroes that work well together)
SYNERGY_PAIRS = [
    (1, 25),   # Ana + Reinhardt
    (7, 16),   # D.Va + Lucio
    (14, 28),  # Kiriko + Sojourn
    (20, 25),  # Moira + Reinhardt
    (35, 16),  # Winston + Lucio
]

# Counter pairs (one hero counters another)
COUNTER_PAIRS = [
    (34, 7),   # Widowmaker counters D.Va
    (7, 34),   # D.Va counters Widowmaker
    (11, 25),  # Hanzo counters Reinhardt
]


def generate_team_composition(
    metadata: any, role_balance_preference: float = 0.7
) -> List[int]:
    """
    Generate a realistic team composition.

    Args:
        metadata: HeroMetadata instance
        role_balance_preference: Probability of generating balanced team (1-2-2)

    Returns:
        List of 5 hero IDs
    """
    # Decide on composition type
    if random.random() < role_balance_preference:
        # Balanced composition: 1 Tank, 2 DPS, 2 Support
        tank = random.choice(metadata.get_heroes_by_role("Tank"))
        dps = random.sample(metadata.get_heroes_by_role("DPS"), 2)
        support = random.sample(metadata.get_heroes_by_role("Support"), 2)
        team = [tank] + dps + support
    else:
        # Random composition (may be unbalanced)
        all_heroes = list(metadata.heroes.keys())
        team = random.sample(all_heroes, 5)

    return team


def calculate_team_strength(team: List[int], metadata: any) -> float:
    """
    Calculate team strength score based on heuristics.

    Args:
        team: List of hero IDs
        metadata: HeroMetadata instance

    Returns:
        Team strength score (0-1)
    """
    score = 0.0

    # Role balance score
    role_dist = metadata.get_role_distribution(team)
    if role_dist["Tank"] == 1 and role_dist["DPS"] == 2 and role_dist["Support"] == 2:
        score += 0.3  # Optimal composition

    # Meta hero bonus
    meta_count = sum(1 for hero_id in team if hero_id in META_HEROES)
    score += (meta_count / 5) * 0.3

    # Synergy bonus
    synergy_count = 0
    for hero1, hero2 in SYNERGY_PAIRS:
        if hero1 in team and hero2 in team:
            synergy_count += 1
    score += min(synergy_count / 2, 0.2) * 0.2

    # Random variation
    score += random.uniform(-0.1, 0.1)

    return max(0.0, min(1.0, score))


def generate_match(
    metadata: any, difficulty_variance: float = 0.2
) -> Tuple[List[int], List[int], int]:
    """
    Generate a single match with teams and outcome.

    Args:
        metadata: HeroMetadata instance
        difficulty_variance: Variance in team strength difference

    Returns:
        Tuple of (team1, team2, winner) where winner is 0 or 1
    """
    team1 = generate_team_composition(metadata)
    team2 = generate_team_composition(metadata)

    # Ensure no overlapping heroes
    while set(team1) & set(team2):
        team2 = generate_team_composition(metadata)

    strength1 = calculate_team_strength(team1, metadata)
    strength2 = calculate_team_strength(team2, metadata)

    # Add variance
    strength1 += random.uniform(-difficulty_variance, difficulty_variance)
    strength2 += random.uniform(-difficulty_variance, difficulty_variance)

    # Determine winner based on strength (with some randomness)
    win_prob_team1 = 1 / (1 + np.exp(-5 * (strength1 - strength2)))
    winner = 1 if random.random() < win_prob_team1 else 0

    return (team1, team2, winner)


def generate_synthetic_data(
    n_matches: int = 10000,
    output_path: Path | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic match dataset.

    Args:
        n_matches: Number of matches to generate
        output_path: Path to save CSV file (optional)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with match data
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    metadata = get_hero_metadata()

    matches = []
    for i in range(n_matches):
        team1, team2, winner = generate_match(metadata)

        match_data = {
            "match_id": i,
            "team1_hero1": team1[0],
            "team1_hero2": team1[1],
            "team1_hero3": team1[2],
            "team1_hero4": team1[3],
            "team1_hero5": team1[4],
            "team2_hero1": team2[0],
            "team2_hero2": team2[1],
            "team2_hero3": team2[2],
            "team2_hero4": team2[3],
            "team2_hero5": team2[4],
            "winner": winner,  # 0 = team2 wins, 1 = team1 wins
        }
        matches.append(match_data)

    df = pd.DataFrame(matches)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {n_matches} matches and saved to {output_path}")

    return df


if __name__ == "__main__":
    # Generate data
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_file = data_dir / "synthetic_matches.csv"
    df = generate_synthetic_data(n_matches=10000, output_path=output_file)
    print(f"\nGenerated dataset shape: {df.shape}")
    print(f"Winner distribution:\n{df['winner'].value_counts()}")

