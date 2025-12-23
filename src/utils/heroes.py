"""Hero metadata and utilities for Overwatch 2."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

HERO_DATA_PATH = Path(__file__).parent.parent.parent / "configs" / "heroes.json"


class HeroMetadata:
    """Hero metadata loader and validator."""

    def __init__(self, heroes_file: Path = HERO_DATA_PATH) -> None:
        """
        Initialize hero metadata from JSON file.

        Args:
            heroes_file: Path to heroes.json file

        Raises:
            FileNotFoundError: If heroes file doesn't exist
            ValueError: If heroes file is invalid
        """
        if not heroes_file.exists():
            raise FileNotFoundError(f"Heroes file not found: {heroes_file}")

        with open(heroes_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.heroes: Dict[int, Dict[str, any]] = {
            hero["id"]: hero for hero in data["heroes"]
        }
        self.roles: Dict[str, List[int]] = data["roles"]

        # Validate data consistency
        all_role_heroes = set()
        for role_heroes in self.roles.values():
            all_role_heroes.update(role_heroes)

        all_hero_ids = set(self.heroes.keys())
        if all_role_heroes != all_hero_ids:
            raise ValueError("Hero IDs in roles don't match hero IDs in heroes list")

    def get_hero(self, hero_id: int) -> Optional[Dict[str, any]]:
        """
        Get hero metadata by ID.

        Args:
            hero_id: Hero ID

        Returns:
            Hero metadata dict or None if not found
        """
        return self.heroes.get(hero_id)

    def get_hero_name(self, hero_id: int) -> Optional[str]:
        """
        Get hero name by ID.

        Args:
            hero_id: Hero ID

        Returns:
            Hero name or None if not found
        """
        hero = self.get_hero(hero_id)
        return hero["name"] if hero else None

    def get_hero_role(self, hero_id: int) -> Optional[str]:
        """
        Get hero role by ID.

        Args:
            hero_id: Hero ID

        Returns:
            Hero role (Tank/DPS/Support) or None if not found
        """
        hero = self.get_hero(hero_id)
        return hero["role"] if hero else None

    def get_heroes_by_role(self, role: str) -> List[int]:
        """
        Get all hero IDs for a given role.

        Args:
            role: Role name (Tank/DPS/Support)

        Returns:
            List of hero IDs
        """
        return self.roles.get(role, [])

    def validate_hero_id(self, hero_id: int) -> bool:
        """
        Validate that hero ID exists.

        Args:
            hero_id: Hero ID to validate

        Returns:
            True if valid, False otherwise
        """
        return hero_id in self.heroes

    def validate_hero_ids(self, hero_ids: List[int]) -> bool:
        """
        Validate that all hero IDs exist.

        Args:
            hero_ids: List of hero IDs to validate

        Returns:
            True if all valid, False otherwise
        """
        return all(self.validate_hero_id(hid) for hid in hero_ids)

    def get_role_distribution(self, hero_ids: List[int]) -> Dict[str, int]:
        """
        Get role distribution for a list of heroes.

        Args:
            hero_ids: List of hero IDs

        Returns:
            Dictionary mapping role names to counts
        """
        distribution = {"Tank": 0, "DPS": 0, "Support": 0}

        for hero_id in hero_ids:
            role = self.get_hero_role(hero_id)
            if role:
                distribution[role] = distribution.get(role, 0) + 1

        return distribution

    def is_valid_team_composition(self, hero_ids: List[int]) -> bool:
        """
        Validate team composition rules.

        Args:
            hero_ids: List of hero IDs (should be 5-6 heroes)

        Returns:
            True if valid composition, False otherwise
        """
        if not (5 <= len(hero_ids) <= 6):
            return False

        if len(hero_ids) != len(set(hero_ids)):
            return False  # Duplicate heroes

        if not self.validate_hero_ids(hero_ids):
            return False  # Invalid hero IDs

        return True


# Global instance
_hero_metadata: Optional[HeroMetadata] = None


def get_hero_metadata() -> HeroMetadata:
    """
    Get global hero metadata instance (singleton).

    Returns:
        HeroMetadata instance
    """
    global _hero_metadata
    if _hero_metadata is None:
        _hero_metadata = HeroMetadata()
    return _hero_metadata

