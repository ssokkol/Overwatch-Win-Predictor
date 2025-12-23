"""Orchestrate data collection and preprocessing."""

from pathlib import Path

import pandas as pd

from src.data.generate_synthetic import generate_synthetic_data
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def collect_and_save_matches(
    output_path: Path,
    n_matches: int = 10000,
    use_synthetic: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Collect match data and save to file.

    Args:
        output_path: Path to save collected data
        n_matches: Number of matches to generate
        use_synthetic: Whether to use synthetic data generation
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with collected matches

    Note:
        Currently only supports synthetic data generation.
        Real API integration can be added later.
    """
    if use_synthetic:
        logger.info(f"Generating {n_matches} synthetic matches")
        df = generate_synthetic_data(
            n_matches=n_matches, output_path=output_path, random_seed=random_seed
        )
    else:
        # Placeholder for real API data collection
        logger.warning("Real API data collection not yet implemented")
        df = generate_synthetic_data(
            n_matches=n_matches, output_path=output_path, random_seed=random_seed
        )

    logger.info(f"Collected {len(df)} matches, saved to {output_path}")

    return df


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_file = data_dir / "synthetic_matches.csv"
    df = collect_and_save_matches(output_file, n_matches=10000)
    print(f"Generated dataset: {df.shape}")

