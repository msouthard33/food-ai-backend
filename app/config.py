"""Application configuration loaded from environment variables."""
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    app_secret_key: str = ""  # REQUIRED in production — set APP_SECRET_KEY env var
    log_level: str = "INFO"
    cors_origins: list[str] = [
        "https://foodai.app",
        "http://localhost:3000",
        # React Native / Expo dev (Metro bundler + Expo DevTools)
        "http://localhost:8081",
        "http://localhost:19006",
    ]

    # Database
    database_url: str = ""  # REQUIRED — set DATABASE_URL env var

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""  # REQUIRED in production — set SUPABASE_JWT_SECRET env var
    supabase_storage_bucket: str = "meal-photos"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM — provider-agnostic config
    llm_provider: str = "gemini"  # "openai" | "anthropic" | "gemini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Admin
    admin_api_key: str = ""  # REQUIRED in production — set ADMIN_API_KEY env var

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "testing"

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Fail fast if critical secrets are missing in production."""
        if self.app_env != "production":
            return self

        errors: list[str] = []

        if not self.app_secret_key:
            errors.append("APP_SECRET_KEY must be set in production")

        if self.app_secret_key in ("change-me-in-production", "secret", "password", "changeme"):
            errors.append(
                f"APP_SECRET_KEY value '{self.app_secret_key}' is a known insecure default — "
                "generate a strong random key: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

        if not self.database_url:
            errors.append("DATABASE_URL must be set in production")

        if "localhost" in (self.database_url or "") or "127.0.0.1" in (self.database_url or ""):
            errors.append("DATABASE_URL points to localhost — use a remote database in production")

        if not self.supabase_jwt_secret:
            errors.append(
                "SUPABASE_JWT_SECRET must be set in production — "
                "without it, any UUID will be accepted as a valid auth token"
            )

        if not self.admin_api_key:
            errors.append("ADMIN_API_KEY must be set in production")

        if errors:
            raise ValueError(
                "Production startup blocked due to insecure configuration:\n  - "
                + "\n  - ".join(errors)
            )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
