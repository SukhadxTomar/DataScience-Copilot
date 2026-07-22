"""OpenRouter adapter for the LLMClient contract.

OpenRouter exposes many models (Claude, GPT, Llama, Gemini) behind an
OpenAI-compatible API. The ``openai`` SDK is used here as a transport
only — nothing outside this module imports it.

Structured output: not every model on OpenRouter supports native schema
enforcement, so this adapter uses a universal approach — the schema is
embedded in the system prompt, the response is validated with Pydantic,
and validation failures are fed back to the model for a bounded number
of retries.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from app.llm.base import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_VALIDATION_RETRIES = 2


class OpenRouterClient:
    """LLMClient implementation backed by OpenRouter."""

    def __init__(self, api_key: str, model: str, timeout: float = 120.0) -> None:
        self._model = model
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=2,
        )

    def complete(self, *, system: str, user: str, schema: type[T]) -> T:
        json_schema = schema.model_json_schema()
        system_with_format = (
            f"{system}\n\n"
            "Respond ONLY with a single JSON object matching this JSON Schema. "
            "No markdown fences, no explanation, no extra text.\n"
            f"Schema:\n{json.dumps(json_schema)}"
        )

        messages = [
            {"role": "system", "content": system_with_format},
            {"role": "user", "content": user},
        ]

        last_error: Exception | None = None
        for attempt in range(1 + MAX_VALIDATION_RETRIES):
            raw = self._call(messages)
            try:
                return schema.model_validate_json(_strip_fences(raw))
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "LLM output failed validation (attempt %d/%d): %s",
                    attempt + 1,
                    1 + MAX_VALIDATION_RETRIES,
                    exc,
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response failed schema validation with this error:\n"
                        f"{exc}\n"
                        "Return ONLY the corrected JSON object."
                    ),
                })

        raise LLMValidationError(
            f"Model failed to produce valid {schema.__name__} after "
            f"{1 + MAX_VALIDATION_RETRIES} attempts"
        ) from last_error

    def _call(self, messages: list[dict]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthError("OpenRouter API key invalid or missing") from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError("OpenRouter rate limit exceeded") from exc
        except openai.APIError as exc:
            raise LLMError(f"OpenRouter API error: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("OpenRouter returned an empty response")
        return content


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that some models wrap around JSON."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else 3
        text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()
