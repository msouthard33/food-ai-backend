"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    app_secret_key: str = "change-me-in-production"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/foodai"

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_storage_bucket: str = "meal-photos"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM — provider-agnostic config
    llm_provider: str = "openai"  # "openai" | "anthropic"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "testing"


@lru_cache
def get_settings() -> Settings:
    return Settings()
