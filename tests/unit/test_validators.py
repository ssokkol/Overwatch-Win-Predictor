"""Unit tests for input validation."""

import pytest

from src.utils.validators import (
    validate_hero_id,
    validate_hero_ids,
    validate_match_request,
    validate_no_duplicates,
    validate_no_overlapping_heroes,
    validate_team_composition,
    validate_team_size,
)


class TestValidators:
    """Test validation utilities."""

    def test_validate_hero_id_valid(self) -> None:
        """Test valid hero ID."""
        assert validate_hero_id(1) is True
        assert validate_hero_id(35) is True

    def test_validate_hero_id_invalid(self) -> None:
        """Test invalid hero ID."""
        assert validate_hero_id(999) is False
        assert validate_hero_id(0) is False

    def test_validate_hero_ids_valid(self) -> None:
        """Test valid hero IDs."""
        assert validate_hero_ids([1, 5, 10, 15, 20]) is True

    def test_validate_hero_ids_invalid(self) -> None:
        """Test invalid hero IDs."""
        assert validate_hero_ids([1, 999, 10]) is False

    def test_validate_team_size_valid(self) -> None:
        """Test valid team size."""
        assert validate_team_size([1, 2, 3, 4, 5]) is True
        assert validate_team_size([1, 2, 3, 4, 5, 6]) is True

    def test_validate_team_size_invalid(self) -> None:
        """Test invalid team size."""
        assert validate_team_size([1, 2, 3]) is False
        assert validate_team_size([1, 2, 3, 4, 5, 6, 7]) is False

    def test_validate_no_duplicates_valid(self) -> None:
        """Test no duplicates."""
        assert validate_no_duplicates([1, 2, 3, 4, 5]) is True

    def test_validate_no_duplicates_invalid(self) -> None:
        """Test duplicates detected."""
        assert validate_no_duplicates([1, 2, 3, 4, 1]) is False

    def test_validate_team_composition_valid(self) -> None:
        """Test valid team composition."""
        assert validate_team_composition([1, 5, 10, 15, 20]) is True

    def test_validate_team_composition_invalid_size(self) -> None:
        """Test invalid team size."""
        with pytest.raises(ValueError, match="Team must have between 5 and 6 heroes"):
            validate_team_composition([1, 2, 3])

    def test_validate_team_composition_duplicates(self) -> None:
        """Test duplicate heroes."""
        with pytest.raises(ValueError, match="Team cannot have duplicate heroes"):
            validate_team_composition([1, 1, 3, 4, 5])

    def test_validate_no_overlapping_heroes_valid(self) -> None:
        """Test no overlapping heroes."""
        assert validate_no_overlapping_heroes([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) is True

    def test_validate_no_overlapping_heroes_invalid(self) -> None:
        """Test overlapping heroes."""
        with pytest.raises(ValueError, match="Heroes cannot appear in both teams"):
            validate_no_overlapping_heroes([1, 2, 3, 4, 5], [5, 6, 7, 8, 9])

    def test_validate_match_request_valid(self) -> None:
        """Test valid match request."""
        team1 = [1, 2, 3, 4, 5]
        team2 = [6, 7, 8, 9, 10]
        is_valid, errors = validate_match_request(team1, team2)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_match_request_invalid(self) -> None:
        """Test invalid match request."""
        team1 = [1, 2, 3]  # Invalid size
        team2 = [5, 6, 7, 8, 9]  # Overlapping with team1
        is_valid, errors = validate_match_request(team1, team2)

        assert is_valid is False
        assert len(errors) > 0

