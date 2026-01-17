"""Main FastAPI application for Overwatch Win Predictor."""

import os
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status

from src.api.dependencies import get_model, get_redis_client, verify_api_key
from src.api.middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    setup_cors_middleware,
    setup_trusted_host_middleware,
)
from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from src.features.team_composition import TeamCompositionFeatureExtractor
from src.models.ensemble import EnsembleModel
from src.utils.heroes import get_hero_metadata
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
environment = os.getenv("ENVIRONMENT", "development")
app = FastAPI(
    title="OverWatch Win Predictor API",
    version="1.0.0",
    description="ML System for Overwatch 2 Match Outcome Prediction",
    docs_url=None if environment == "production" else "/docs",
    redoc_url=None if environment == "production" else "/redoc",
)

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)

redis_client = get_redis_client()
app.add_middleware(
    RateLimitMiddleware,
    redis_client=redis_client,
    calls=int(os.getenv("RATE_LIMIT_CALLS", "100")),
    period=int(os.getenv("RATE_LIMIT_PERIOD", "60")),
)

setup_cors_middleware(app)
setup_trusted_host_middleware(app)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status and version
    """
    return HealthResponse(status="healthy", version="1.0.0")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Prediction"],
)
async def predict_match(
    request: PredictionRequest,
    model: EnsembleModel = Depends(get_model),
) -> PredictionResponse:
    """
    Predict match outcome given team compositions.

    Args:
        request: Prediction request with team compositions
        model: Trained ensemble model

    Returns:
        Prediction response with win probabilities

    Raises:
        HTTPException: If prediction fails
    """
    try:
        team1 = request.team1.hero_ids
        team2 = request.team2.hero_ids

        logger.info(f"Prediction requested: Team1={team1}, Team2={team2}")

        # Load embeddings if available
        embeddings_path = Path("data/embeddings/hero_embeddings.npy")
        embeddings = None
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)

        # Extract features
        feature_extractor = TeamCompositionFeatureExtractor(embeddings=embeddings)
        feature_vector = feature_extractor.extract_feature_vector(team1, team2)

        # Reshape for single prediction
        X = feature_vector.reshape(1, -1)

        # Predict
        proba = model.predict_proba(X)[0]
        team1_prob = float(proba[1])  # Class 1 = team1 wins
        team2_prob = float(proba[0])  # Class 0 = team2 wins

        # Determine winner
        predicted_winner = "Team1" if team1_prob > team2_prob else "Team2"
        confidence = max(team1_prob, team2_prob)

        return PredictionResponse(
            team1_win_probability=team1_prob,
            team2_win_probability=team2_prob,
            predicted_winner=predicted_winner,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/recommendations",
    response_model=RecommendationResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Recommendations"],
)
async def get_recommendations(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """
    Get hero recommendations based on current and enemy team compositions.

    Args:
        request: Recommendation request with team compositions

    Returns:
        Recommended heroes with reasoning

    Note:
        This is a simplified implementation. In production, use ML-based
        recommendation system.
    """
    try:
        metadata = get_hero_metadata()
        current_team = set(request.current_team.hero_ids)
        enemy_team = set(request.enemy_team.hero_ids)

        # Simple heuristic: recommend heroes not in either team
        all_heroes = set(metadata.heroes.keys())
        available_heroes = all_heroes - current_team - enemy_team

        # Get role distribution of current team
        role_dist = metadata.get_role_distribution(list(current_team))

        # Prioritize heroes that balance the team
        recommended = []
        if role_dist["Tank"] < 1:
            tank_heroes = [
                h for h in metadata.get_heroes_by_role("Tank") if h in available_heroes
            ]
            recommended.extend(tank_heroes[:1])

        if role_dist["DPS"] < 2:
            dps_heroes = [
                h for h in metadata.get_heroes_by_role("DPS") if h in available_heroes
            ]
            recommended.extend(dps_heroes[: (2 - role_dist["DPS"])])

        if role_dist["Support"] < 2:
            support_heroes = [
                h
                for h in metadata.get_heroes_by_role("Support")
                if h in available_heroes
            ]
            recommended.extend(support_heroes[: (2 - role_dist["Support"])])

        # Fill remaining slots with any available heroes
        remaining_needed = request.num_recommendations - len(recommended)
        if remaining_needed > 0:
            other_heroes = [
                h for h in available_heroes if h not in recommended
            ][:remaining_needed]
            recommended.extend(other_heroes)

        reasoning = (
            "Recommended heroes to balance team composition and provide strong "
            "counter-picks."
        )

        return RecommendationResponse(
            recommended_heroes=recommended[: request.num_recommendations],
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {str(e)}",
        )


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize application on startup."""
    logger.info("Starting OverWatch Win Predictor API")
    logger.info(f"Environment: {environment}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up on shutdown."""
    logger.info("Shutting down OverWatch Win Predictor API")


def main() -> None:
    """Run the application."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
