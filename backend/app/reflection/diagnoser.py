"""Reflection agent — turns a capability failure into a structured Diagnosis.

Two-tier by design and by cost:

1. Deterministic heuristics (``taxonomy.classify``) classify the exception by
   type and message. Fast, free, and covering the known failure modes — the
   XGBoost target-encoding case is a pure heuristic hit that never calls the LLM.
2. Only when the heuristic is *low-confidence* (``< min_confidence``) does the
   agent fall back to the LLM, which must return a schema-valid ``Diagnosis``
   (reusing the same ``LLMClient`` structured-output contract every other agent
   uses).

Robustness guarantees:

- Unknown exceptions become a low-confidence ``UNKNOWN_ERROR`` from the taxonomy,
  which routes to the LLM rather than being accepted as a terminal verdict.
- The LLM fallback runs under ``diagnosis_timeout_s`` in a worker thread. A
  hanging provider must never stall the graph: on timeout — or any ``LLMError``,
  or a category the model invents outside the taxonomy — the agent returns the
  heuristic diagnosis it already has. Diagnosis is best-effort; a slow or broken
  provider degrades to the deterministic verdict, it does not block the run.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings
from app.llm.base import LLMClient, LLMError
from app.reflection.models import Diagnosis, FailureCategory
from app.reflection.taxonomy import HEURISTIC_MIN_CONFIDENCE, classify

logger = logging.getLogger(__name__)


class _LLMDiagnosis(BaseModel):
    """Constrained shape the LLM must return. Kept separate from the internal
    ``Diagnosis`` so the model only chooses fields it is competent to choose;
    the agent maps it onto a full ``Diagnosis`` with ``source='llm'``."""

    category: FailureCategory = Field(
        description="One of the fixed failure categories that best fits the error"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Root cause in 1-2 sentences")
    suggested_repair: str | None = Field(
        default=None,
        description="Name of an artifact-level repair to try, or null if the fix "
        "is a planning decision (model/config/resource)",
    )


_SYSTEM_PROMPT = """You are the reflection agent for an automated machine-learning \
pipeline. A pipeline step (capability) raised an exception. Diagnose the ROOT \
CAUSE and classify it into exactly one fixed category.

Categories:
- DATA_SCHEMA_ERROR — feature/column mismatch, parsing, unexpected structure
- TARGET_ENCODING_ERROR — the target label is non-numeric / needs encoding
- MISSING_VALUES — NaNs or infinities where complete data is required
- TYPE_ERROR — dtype mismatch, a numeric column holding non-numeric values
- MODEL_CONFIGURATION_ERROR — the model/CV setup is invalid for this data
- RESOURCE_LIMIT — out of memory / resource exhaustion
- TRANSIENT_ERROR — rate limit, timeout, temporary provider/network failure
- VALIDATION_ERROR — empty or degenerate input failed a basic check
- UNKNOWN_ERROR — none of the above fit

Only suggest a `suggested_repair` for an ARTIFACT-level fix (encoding a column, \
imputing, dropping rows/columns, realigning schema, retrying a transient call). \
If the right response is to change the modeling strategy (swap the model, reduce \
the search space), leave `suggested_repair` null — that is a planner decision, \
not a repair."""


class ReflectionAgent:
    """Produces a :class:`Diagnosis` from a failed capability's exception."""

    def __init__(
        self,
        llm: LLMClient,
        min_confidence: float = HEURISTIC_MIN_CONFIDENCE,
        diagnosis_timeout_s: float | None = None,
    ) -> None:
        self._llm = llm
        self._min_confidence = min_confidence
        self._timeout_s = (
            diagnosis_timeout_s
            if diagnosis_timeout_s is not None
            else settings.diagnosis_timeout_s
        )

    def diagnose(
        self,
        *,
        failed_capability: str,
        error: str,
        exc_type: str,
        state: dict[str, Any] | None = None,
    ) -> Diagnosis:
        """Diagnose one failure. Heuristic first; LLM fallback if low-confidence.

        Never raises: a provider failure or timeout degrades to the heuristic
        verdict so the reflection loop always makes forward progress.
        """
        heuristic = classify(exc_type, error)
        if heuristic.confidence >= self._min_confidence:
            logger.info(
                "reflect: heuristic diagnosis of '%s' -> %s (conf=%.2f)",
                failed_capability, heuristic.category.value, heuristic.confidence,
            )
            return heuristic

        logger.info(
            "reflect: heuristic low-confidence (%.2f) for '%s' — consulting LLM",
            heuristic.confidence, failed_capability,
        )
        llm = self._llm_diagnose(
            failed_capability=failed_capability, error=error, exc_type=exc_type
        )
        return llm if llm is not None else heuristic

    def _llm_diagnose(
        self, *, failed_capability: str, error: str, exc_type: str
    ) -> Diagnosis | None:
        """Run the LLM fallback under a wall-clock timeout. Returns None on any
        failure (timeout, provider error, or unusable output) so the caller can
        fall back to the heuristic."""
        user = (
            f"Failed capability: {failed_capability}\n"
            f"Exception type: {exc_type}\n"
            f"Exception message:\n{error}"
        )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self._llm.complete,
                    system=_SYSTEM_PROMPT,
                    user=user,
                    schema=_LLMDiagnosis,
                )
                out = future.result(timeout=self._timeout_s)
        except FutureTimeout:
            logger.warning(
                "reflect: LLM diagnosis timed out after %.1fs — using heuristic",
                self._timeout_s,
            )
            return None
        except LLMError as exc:
            logger.warning("reflect: LLM diagnosis failed (%s) — using heuristic", exc)
            return None
        except Exception as exc:  # noqa: BLE001 — diagnosis must never propagate
            logger.warning(
                "reflect: unexpected LLM diagnosis error (%s) — using heuristic", exc
            )
            return None

        return Diagnosis(
            category=out.category,
            confidence=out.confidence,
            reason=out.reason,
            suggested_repair=out.suggested_repair,
            source="llm",
        )
