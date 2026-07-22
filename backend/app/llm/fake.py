"""In-memory LLM client for tests and offline development.

Returns pre-programmed responses instead of making network calls,
keeping tests fast, free, and deterministic.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.llm.base import LLMError

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    """LLMClient implementation that replays canned responses.

    Example:
        fake = FakeLLMClient(responses=[InsightReport(...)])
        agent = EDAInsightsAgent(llm=fake)
    """

    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, schema: type[T]) -> T:
        self.calls.append({"system": system, "user": user, "schema": schema.__name__})
        if not self._responses:
            raise LLMError("FakeLLMClient ran out of canned responses")
        response = self._responses.pop(0)
        if not isinstance(response, schema):
            raise LLMError(
                f"FakeLLMClient response type {type(response).__name__} "
                f"does not match requested schema {schema.__name__}"
            )
        return response
