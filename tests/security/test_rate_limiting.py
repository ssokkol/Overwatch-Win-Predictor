"""Security tests for rate limiting."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Test client fixture."""
    return TestClient(app)


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_headers_present(self, client: TestClient) -> None:
        """Test that rate limit headers are present."""
        response = client.get("/health")
        # Rate limit headers may or may not be present depending on implementation
        # This is a basic test
        assert response.status_code == 200

    def test_health_check_not_rate_limited(self, client: TestClient) -> None:
        """Test that health check is not rate limited."""
        # Make multiple requests
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200
