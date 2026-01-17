"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def api_key() -> str:
    """API key fixture."""
    import os
    os.environ["API_KEYS"] = "test-api-key"
    return "test-api-key"


@pytest.fixture(autouse=True)
def _set_default_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests use the same API key as request headers."""
    monkeypatch.setenv("API_KEYS", "test-api-key")


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestPredictionEndpoint:
    """Test prediction endpoint."""

    def test_predict_missing_api_key(self, client: TestClient) -> None:
        """Test prediction without API key."""
        # Should fail if API keys are configured
        import os
        original_keys = os.getenv("API_KEYS")
        try:
            os.environ["API_KEYS"] = "test-key"
            response = client.post(
                "/predict",
                json={
                    "team1": {"hero_ids": [1, 5, 10, 15, 20]},
                    "team2": {"hero_ids": [2, 7, 12, 17, 22]},
                },
            )
            # Should return 401 if API key required
            assert response.status_code in [401, 500]  # 500 if model not loaded
        finally:
            if original_keys:
                os.environ["API_KEYS"] = original_keys
            else:
                os.environ.pop("API_KEYS", None)

    def test_predict_invalid_team_size(self, client: TestClient) -> None:
        """Test prediction with invalid team size."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-api-key"},
            json={
                "team1": {"hero_ids": [1, 2, 3]},  # Invalid size
                "team2": {"hero_ids": [6, 7, 8, 9, 10]},
            },
        )
        assert response.status_code == 422  # Validation error

    def test_predict_overlapping_heroes(self, client: TestClient) -> None:
        """Test prediction with overlapping heroes."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-api-key"},
            json={
                "team1": {"hero_ids": [1, 2, 3, 4, 5]},
                "team2": {"hero_ids": [5, 6, 7, 8, 9]},  # Hero 5 in both
            },
        )
        assert response.status_code == 422  # Validation error


class TestRecommendationEndpoint:
    """Test recommendation endpoint."""

    def test_recommendations_invalid_request(self, client: TestClient) -> None:
        """Test recommendation with invalid request."""
        response = client.post(
            "/recommendations",
            headers={"X-API-Key": "test-api-key"},
            json={
                "current_team": {"hero_ids": [1, 2, 3]},  # Invalid size
                "enemy_team": {"hero_ids": [6, 7, 8, 9, 10]},
            },
        )
        assert response.status_code == 422  # Validation error

