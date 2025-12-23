"""Application configuration and settings."""

import os
from functools import lru_cache
from typing import List

from pydantic import SecretStr

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore


class Settings(BaseSettings):
    """Application settings with secure secret handling."""

    # Environment
    environment: str = "development"
    debug: bool = False

    # API Configuration
    api_keys: SecretStr = SecretStr("")
    secret_key: SecretStr = SecretStr("")
    allowed_origins: str = "http://localhost,http://localhost:3000"
    allowed_hosts: str = "localhost,127.0.0.1"

    # Rate Limiting
    rate_limit_calls: int = 100
    rate_limit_period: int = 60

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: SecretStr = SecretStr("")

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Model
    model_dir: str = "models/ensemble"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False

    def get_api_keys(self) -> List[str]:
        """
        Get API keys as list.

        Returns:
            List of API keys
        """
        keys_str = self.api_keys.get_secret_value()
        if not keys_str:
            return []
        return [key.strip() for key in keys_str.split(",") if key.strip()]

    def get_allowed_origins(self) -> List[str]:
        """
        Get allowed origins as list.

        Returns:
            List of allowed origins
        """
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    def get_allowed_hosts(self) -> List[str]:
        """
        Get allowed hosts as list.

        Returns:
            List of allowed hosts
        """
        return [host.strip() for host in self.allowed_hosts.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton).

    Returns:
        Settings instance
    """
    return Settings()

