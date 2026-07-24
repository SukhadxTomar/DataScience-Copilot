"""Application settings, loaded from environment variables or a .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    # Storage
    data_dir: Path = Path("storage")

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Upload limits
    max_upload_mb: int = 200

    # LLM
    llm_provider: str = "openrouter"
    llm_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_api_key: str = ""

    # Planning — bounded retries (see app/planner)
    plan_attempts: int = 3     # tries for the planner to produce a valid plan
    replan_attempts: int = 2   # runtime replans allowed when a capability fails

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"


settings = Settings()
settings.datasets_dir.mkdir(parents=True, exist_ok=True)
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
