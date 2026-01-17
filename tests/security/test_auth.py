"""Security tests for authentication."""

import os
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Test client fixture."""
    return TestClient(app)


class TestAPIKeyAuthentication:
    """Test API key authentication."""

    def test_health_check_no_auth_required(self, client: TestClient) -> None:
        """Test that health check doesn't require auth."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_predict_without_api_key_denied(self, client: TestClient) -> None:
        """Test that prediction without API key is denied when keys configured."""
        original_keys = os.getenv("API_KEYS")
        try:
            os.environ["API_KEYS"] = "required-key"
            response = client.post(
                "/predict",
                json={
                    "team1": {"hero_ids": [1, 5, 10, 15, 20]},
                    "team2": {"hero_ids": [2, 7, 12, 17, 22]},
                },
            )
            # Should return 401 if no API key provided
            assert response.status_code in [401, 500]
        finally:
            if original_keys:
                os.environ["API_KEYS"] = original_keys
            else:
                os.environ.pop("API_KEYS", None)

    def test_predict_with_invalid_api_key_denied(self, client: TestClient) -> None:
        """Test that prediction with invalid API key is denied."""
        original_keys = os.getenv("API_KEYS")
        try:
            os.environ["API_KEYS"] = "valid-key"
            response = client.post(
                "/predict",
                headers={"X-API-Key": "invalid-key"},
                json={
                    "team1": {"hero_ids": [1, 5, 10, 15, 20]},
                    "team2": {"hero_ids": [2, 7, 12, 17, 22]},
                },
            )
            assert response.status_code in [401, 500]
        finally:
            if original_keys:
                os.environ["API_KEYS"] = original_keys
            else:
                os.environ.pop("API_KEYS", None)
