"""Step 5.1: the API's settings, read from environment variables (or a local .env file).

Every setting can be set as MKOPOGUARD_<NAME>, e.g. MKOPOGUARD_SECRET_KEY. Locally they come
from .env in the project root, which is git-ignored and never committed; .env.example lists
every setting with a safe placeholder. On the host (Phase 8) they are set in its dashboard.

The app refuses to start with missing or unsafe settings, instead of failing later, on the
first request, in a way a user would see.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mkopoguard import config

ENV_FILE = config.PROJECT_ROOT / ".env"
ENV_PREFIX = "MKOPOGUARD_"
PLACEHOLDER = "change-me"


class DatabaseSettings(BaseSettings):
    """Only the database URL: what Alembic needs to run migrations (no secret key required)."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = f"sqlite:///{(config.PROJECT_ROOT / 'mkopoguard.db').as_posix()}"


class Settings(DatabaseSettings):
    environment: Literal["development", "test", "production"] = "development"
    # Signs the login tokens (step 5.3). Anyone who has it can forge a login, so it is a secret.
    secret_key: SecretStr = Field(min_length=32)
    access_token_minutes: int = Field(default=30, ge=5, le=120)
    bundle_dir: Path = config.BUNDLE_DIR
    frontend_dir: Path = config.PROJECT_ROOT / "frontend"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Login rate limits (step 5.7)
    login_attempts_per_ip: int = Field(default=20, ge=1)
    login_ip_window_seconds: int = Field(default=300, ge=1)
    login_failures_per_account: int = Field(default=5, ge=1)
    login_failure_window_seconds: int = Field(default=900, ge=1)

    @field_validator("secret_key")
    @classmethod
    def _not_the_placeholder(cls, value: SecretStr) -> SecretStr:
        if PLACEHOLDER in value.get_secret_value():
            raise ValueError("secret_key is still the placeholder from .env.example; generate one")
        return value

    @field_validator("frontend_dir")
    @classmethod
    def _frontend_exists(cls, value: Path) -> Path:
        if not (value / "index.html").exists():
            raise ValueError(f"No index.html in the frontend folder {value}")
        return value

    @model_validator(mode="after")
    def _production_needs_a_real_database(self) -> "Settings":
        if self.environment == "production" and self.database_url.startswith("sqlite"):
            raise ValueError("Production needs a hosted database (Supabase Postgres, step 8.2)")
        return self


@lru_cache
def get_settings() -> Settings:
    """The settings, read once per process."""
    return Settings()
