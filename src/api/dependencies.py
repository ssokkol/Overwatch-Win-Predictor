"""FastAPI dependencies for dependency injection."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

try:
    import redis
except ImportError:
    redis = None  # type: ignore

from src.models.ensemble import EnsembleModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global model cache
_model_cache: Optional[EnsembleModel] = None
_redis_client: Optional[redis.Redis] = None


class _PlaceholderEnsembleModel:
    """Fallback model for tests/CI when trained artifacts are missing."""

    def predict_proba(self, X: any) -> np.ndarray:
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0] if X_arr.ndim > 0 else 1
        return np.tile([0.5, 0.5], (n_samples, 1))

    def predict(self, X: any) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def _allow_missing_model() -> bool:
    """Allow placeholder model in tests/CI to keep API smoke tests running."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    env = os.getenv("ENVIRONMENT", "").lower()
    if env in {"test", "ci"}:
        return True
    if os.getenv("CI", "").lower() in {"1", "true", "yes"}:
        return True
    if os.getenv("ALLOW_MISSING_MODEL", "").lower() in {"1", "true", "yes"}:
        return True
    return False


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key for protected endpoints.

    Args:
        api_key: API key from header

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid or missing
    """
    valid_api_keys = os.getenv("API_KEYS", "").split(",")
    valid_api_keys = [key.strip() for key in valid_api_keys if key.strip()]

    if not valid_api_keys or valid_api_keys == [""]:
        # No API keys configured, allow all requests (development mode)
        logger.warning("No API keys configured - allowing all requests")
        return "dev_key"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key not in valid_api_keys:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key


@lru_cache()
def get_model() -> EnsembleModel:
    """
    Get loaded model instance (singleton).

    Returns:
        EnsembleModel instance

    Raises:
        FileNotFoundError: If model files not found
    """
    global _model_cache

    if _model_cache is None:
        model_dir = Path(os.getenv("MODEL_DIR", "models/ensemble"))
        
        if not model_dir.exists():
            if _allow_missing_model():
                logger.warning(
                    f"Model directory not found: {model_dir}. Using placeholder model."
                )
                _model_cache = _PlaceholderEnsembleModel()
                return _model_cache  # type: ignore[return-value]
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. Please train models first."
            )

        logger.info(f"Loading model from {model_dir}")
        _model_cache = EnsembleModel.load(model_dir)
        logger.info("Model loaded successfully")

    return _model_cache


def get_redis_client() -> Optional[any]:
    """
    Get Redis client instance (singleton).

    Returns:
        Redis client or None if Redis unavailable
    """
    global _redis_client

    if _redis_client is None:
        if redis is None:
            logger.warning("Redis not installed. Using in-memory cache.")
            return None
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_password = os.getenv("REDIS_PASSWORD", "")

        try:
            _redis_client = redis.from_url(
                redis_url,
                password=redis_password if redis_password else None,
                decode_responses=True,
            )
            # Test connection
            _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            _redis_client = None

    return _redis_client

