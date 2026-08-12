"""Application settings, loaded from environment variables or a .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    # Storage
    data_dir: Path = Path("storage")

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000","https://data-science-copilot-red.vercel.app"]

    # Upload limits
    max_upload_mb: int = 200

    # LLM
    llm_provider: str = "openrouter"
    llm_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_api_key: str = ""

    # Planning — bounded retries (see app/planner)
    plan_attempts: int = 3     # tries for the planner to produce a valid plan
    replan_attempts: int = 2   # runtime replans allowed when a capability fails

    # Reflection + auto-fix — bounded retries (see app/reflection)
    reflection_attempts: int = 2   # reflect cycles allowed per failing capability
    repair_attempts: int = 4       # total repairs applied across a whole run
    diagnosis_timeout_s: float = 20.0  # wall-clock cap on the LLM diagnosis fallback

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"


settings = Settings()
settings.datasets_dir.mkdir(parents=True, exist_ok=True)
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
