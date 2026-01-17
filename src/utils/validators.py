"""Input validation utilities."""

from typing import List

from src.utils.heroes import get_hero_metadata


def validate_hero_id(hero_id: int) -> bool:
    """
    Validate that a hero ID is in valid range and exists.

    Args:
        hero_id: Hero ID to validate

    Returns:
        True if valid, False otherwise
    """
    metadata = get_hero_metadata()
    return metadata.validate_hero_id(hero_id)


def validate_hero_ids(hero_ids: List[int]) -> bool:
    """
    Validate that all hero IDs are valid.

    Args:
        hero_ids: List of hero IDs to validate

    Returns:
        True if all valid, False otherwise
    """
    if not hero_ids:
        return False

    metadata = get_hero_metadata()
    return metadata.validate_hero_ids(hero_ids)


def validate_team_size(team: List[int], min_size: int = 5, max_size: int = 6) -> bool:
    """
    Validate team size is within acceptable range.

    Args:
        team: List of hero IDs
        min_size: Minimum team size
        max_size: Maximum team size

    Returns:
        True if team size is valid, False otherwise
    """
    return min_size <= len(team) <= max_size


def validate_no_duplicates(team: List[int]) -> bool:
    """
    Validate team has no duplicate heroes.

    Args:
        team: List of hero IDs

    Returns:
        True if no duplicates, False otherwise
    """
    return len(team) == len(set(team))


def validate_team_composition(team: List[int]) -> bool:
    """
    Validate complete team composition rules.

    Args:
        team: List of hero IDs

    Returns:
        True if valid composition, False otherwise

    Raises:
        ValueError: If validation fails with specific error message
    """
    if not validate_team_size(team):
        raise ValueError(
            f"Team must have between 5 and 6 heroes, got {len(team)}"
        )

    if not validate_no_duplicates(team):
        raise ValueError("Team cannot have duplicate heroes")

    if not validate_hero_ids(team):
        invalid_ids = [hid for hid in team if not validate_hero_id(hid)]
        raise ValueError(f"Invalid hero IDs: {invalid_ids}")

    return True


def validate_no_overlapping_heroes(team1: List[int], team2: List[int]) -> bool:
    """
    Validate that teams don't share heroes.

    Args:
        team1: First team hero IDs
        team2: Second team hero IDs

    Returns:
        True if no overlap, False otherwise

    Raises:
        ValueError: If teams share heroes
    """
    set1 = set(team1)
    set2 = set(team2)
    overlap = set1 & set2

    if overlap:
        raise ValueError(
            f"Heroes cannot appear in both teams: {list(overlap)}"
        )

    return True


def validate_match_request(
    team1: List[int], team2: List[int]
) -> tuple[bool, List[str]]:
    """
    Validate complete match prediction request.

    Args:
        team1: First team hero IDs
        team2: Second team hero IDs

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors: List[str] = []

    try:
        validate_team_composition(team1)
    except ValueError as e:
        errors.append(f"Team 1: {str(e)}")

    try:
        validate_team_composition(team2)
    except ValueError as e:
        errors.append(f"Team 2: {str(e)}")

    try:
        validate_no_overlapping_heroes(team1, team2)
    except ValueError as e:
        errors.append(str(e))

    return (len(errors) == 0, errors)
