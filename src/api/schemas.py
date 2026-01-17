"""Pydantic schemas for API request/response validation."""

from typing import List

from pydantic import BaseModel, Field, validator


class TeamComposition(BaseModel):
    """Schema for team composition with strict validation."""

    hero_ids: List[int] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="List of exactly 5 hero IDs",
        example=[1, 5, 10, 15, 20],
    )

    @validator("hero_ids")
    def validate_hero_ids(cls, v: List[int]) -> List[int]:
        """
        Validate hero IDs are in valid range and unique.

        Args:
            v: List of hero IDs

        Returns:
            Validated hero IDs

        Raises:
            ValueError: If hero IDs are invalid
        """
        if len(set(v)) != len(v):
            raise ValueError("Hero IDs must be unique")

        # Hero IDs should be between 1 and 40 (based on heroes.json)
        if not all(1 <= hero_id <= 40 for hero_id in v):
            raise ValueError("Hero IDs must be between 1 and 40")

        return v

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "hero_ids": [1, 5, 10, 15, 20],
            }
        }


class PredictionRequest(BaseModel):
    """Request model for match prediction."""

    team1: TeamComposition
    team2: TeamComposition

    @validator("team2")
    def validate_no_overlapping_heroes(
        cls, v: TeamComposition, values: dict
    ) -> TeamComposition:
        """
        Ensure no hero appears in both teams.

        Args:
            v: Team 2 composition
            values: Previously validated values (includes team1)

        Returns:
            Validated team 2 composition

        Raises:
            ValueError: If heroes overlap between teams
        """
        if "team1" in values:
            team1_heroes = set(values["team1"].hero_ids)
            team2_heroes = set(v.hero_ids)

            overlap = team1_heroes & team2_heroes
            if overlap:
                raise ValueError(
                    f"Heroes cannot appear in both teams: {list(overlap)}"
                )

        return v

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "team1": {"hero_ids": [1, 5, 10, 15, 20]},
                "team2": {"hero_ids": [2, 7, 12, 17, 22]},
            }
        }


class PredictionResponse(BaseModel):
    """Response model for match prediction."""

    team1_win_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of team1 winning"
    )
    team2_win_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of team2 winning"
    )
    predicted_winner: str = Field(
        ..., description="Predicted winning team ('Team1' or 'Team2')"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of prediction"
    )

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "team1_win_probability": 0.65,
                "team2_win_probability": 0.35,
                "predicted_winner": "Team1",
                "confidence": 0.65,
            }
        }


class RecommendationRequest(BaseModel):
    """Request model for hero recommendations."""

    current_team: TeamComposition
    enemy_team: TeamComposition
    num_recommendations: int = Field(
        default=5, ge=1, le=10, description="Number of recommendations"
    )

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "current_team": {"hero_ids": [1, 5, 10, 15, 20]},
                "enemy_team": {"hero_ids": [2, 7, 12, 17, 22]},
                "num_recommendations": 5,
            }
        }


class RecommendationResponse(BaseModel):
    """Response model for hero recommendations."""

    recommended_heroes: List[int] = Field(
        ..., description="List of recommended hero IDs"
    )
    reasoning: str = Field(..., description="Brief explanation of recommendations")

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "recommended_heroes": [3, 8, 11, 16, 21],
                "reasoning": "These heroes provide strong counters and team synergy",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
            }
        }
