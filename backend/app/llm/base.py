"""Provider-independent LLM client contract.

Agents depend on this interface only. Swapping providers means writing a
new adapter; agent code stays untouched.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for all LLM-related failures."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited the request. Retryable."""


class LLMAuthError(LLMError):
    """Invalid or missing API key. Not retryable."""


class LLMValidationError(LLMError):
    """The model could not produce schema-conformant output."""


class LLMClient(Protocol):
    """Structured completion interface.

    A single method by design: agents always need structured data
    (problem specs, execution plans, insight reports), so free-text
    completions are intentionally not part of the contract.
    """

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:
        """Run a completion and return a validated instance of ``schema``.

        Raises:
            LLMValidationError: the model failed to produce valid JSON
                after retries.
            LLMRateLimitError: the provider rate-limited the request.
            LLMAuthError: the API key is invalid or missing.
            LLMError: any other provider failure.
        """
        ...
