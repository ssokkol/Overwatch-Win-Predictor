"""Data preprocessing utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_match_data(data_path: Path) -> pd.DataFrame:
    """
    Load match data from CSV file.

    Args:
        data_path: Path to CSV file

    Returns:
        DataFrame with match data

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} matches from {data_path}")

    return df


def extract_team_compositions(
    df: pd.DataFrame,
) -> Tuple[list[list[int]], list[list[int]], NDArray[np.int_]]:
    """
    Extract team compositions from DataFrame.

    Args:
        df: DataFrame with match data

    Returns:
        Tuple of (team1_list, team2_list, winners)
    """
    team1_list = []
    team2_list = []
    winners = []

    for _, row in df.iterrows():
        team1 = [
            int(row[f"team1_hero{i}"])
            for i in range(1, 6)
            if f"team1_hero{i}" in row and pd.notna(row[f"team1_hero{i}"])
        ]
        team2 = [
            int(row[f"team2_hero{i}"])
            for i in range(1, 6)
            if f"team2_hero{i}" in row and pd.notna(row[f"team2_hero{i}"])
        ]

        if len(team1) == 5 and len(team2) == 5 and "winner" in row:
            team1_list.append(team1)
            team2_list.append(team2)
            winners.append(int(row["winner"]))

    winners_array = np.array(winners, dtype=np.int_)
    logger.info(f"Extracted {len(team1_list)} valid matches")

    return team1_list, team2_list, winners_array


def train_val_test_split(
    X: NDArray[np.float_],
    y: NDArray[np.int_],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.int_],
    NDArray[np.int_],
    NDArray[np.int_],
]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature matrix
        y: Target labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )

    # Second split: val vs test
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_size_adjusted),
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        f"Split data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate match data.

    Args:
        df: Raw match DataFrame

    Returns:
        Cleaned DataFrame

    Note:
        Removes rows with missing values or invalid team compositions.
    """
    original_len = len(df)
    # Remove rows with missing winner
    df = df.dropna(subset=["winner"])
    # Remove rows with invalid team compositions
    valid_rows = []
    for idx, row in df.iterrows():
        team1_heroes = [row.get(f"team1_hero{i}") for i in range(1, 6)]
        team2_heroes = [row.get(f"team2_hero{i}") for i in range(1, 6)]

        if all(pd.notna(h) for h in team1_heroes) and all(
            pd.notna(h) for h in team2_heroes
        ):
            valid_rows.append(idx)

    df = df.loc[valid_rows]

    removed = original_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} invalid rows")

    return df.reset_index(drop=True)
