"""Application Configuration using Pydantic Settings."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables and .env file."""

    # Project Information
    PROJECT_NAME: str = "SignalScope Backend"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Logging Settings
    LOG_LEVEL: str = "INFO"

    # CORS Settings (allowing frontend communication)
    # Safe default: specific localhost origins for development.
    # In production, set this to your actual frontend domain(s) via .env.
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Image Upload & Validation Settings
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_DIMENSION: int = 8192
    ALLOWED_IMAGE_FORMATS: List[str] = ["JPEG", "PNG", "WEBP"]
    # Hard cap on bytes read from an upload before any image decoding.
    # Must be >= MAX_IMAGE_SIZE_MB; the extra headroom covers multipart overhead.
    MAX_REQUEST_BODY_SIZE_MB: int = 12

    @property
    def max_image_size_bytes(self) -> int:
        """Returns the maximum image size converted to bytes."""
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024

    @property
    def max_request_body_size_bytes(self) -> int:
        """Returns the hard body-read cap in bytes (used to stop reading early on huge uploads)."""
        return self.MAX_REQUEST_BODY_SIZE_MB * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton instance of settings to import across the app
settings = Settings()
