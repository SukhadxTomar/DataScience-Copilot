"""LLM client factory.

Reads provider configuration from settings and returns the matching
client. Agents receive the client through dependency injection and never
construct one themselves.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.llm.base import LLMClient
from app.llm.openrouter_client import OpenRouterClient


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return the configured LLM client (one instance per process)."""
    provider = settings.llm_provider.lower()

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "APP_OPENROUTER_API_KEY is not set. Add it to backend/.env"
            )
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            model=settings.llm_model,
        )

    raise RuntimeError(f"Unknown LLM provider: {provider!r}")
