"""Security middleware for FastAPI application."""

import os
from typing import Callable

from fastapi import HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.api.rate_limiter import RateLimiter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response with security headers
        """
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware per IP address."""

    def __init__(
        self,
        app: any,
        redis_client: any = None,
        calls: int = 100,
        period: int = 60,
    ) -> None:
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            redis_client: Redis client (optional)
            calls: Maximum calls per period
            period: Time period in seconds
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(
            redis_client=redis_client, max_calls=calls, period=period
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limit before processing request.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        is_allowed, remaining = self.rate_limiter.is_allowed(client_ip)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
                headers={"X-RateLimit-Remaining": "0"},
            )

        # Add remaining calls to response headers
        response = await call_next(request)
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


def setup_cors_middleware(app: any) -> None:
    """
    Set up CORS middleware.

    Args:
        app: FastAPI application
    """
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost,http://localhost:3000"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def setup_trusted_host_middleware(app: any) -> None:
    """
    Set up trusted host middleware.

    Args:
        app: FastAPI application
    """
    allowed_hosts_env = os.getenv("ALLOWED_HOSTS")
    if allowed_hosts_env:
        allowed_hosts = [host.strip() for host in allowed_hosts_env.split(",") if host.strip()]
    else:
        allowed_hosts = ["localhost", "127.0.0.1"]

    # Allow FastAPI TestClient host during pytest runs.
    try:
        import sys

        if "pytest" in sys.modules and "testserver" not in allowed_hosts:
            allowed_hosts.append("testserver")
    except Exception:
        pass

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts,
    )
