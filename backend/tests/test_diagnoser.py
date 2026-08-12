"""Tests for the ReflectionAgent's two-tier diagnosis (Phase C).

The agent is heuristic-first: a confident taxonomy match never touches the LLM;
only a low-confidence heuristic falls back to the model. The fallback is
best-effort — a timeout, an LLMError, or an unusable response must degrade to the
heuristic verdict rather than raise, so the reflection loop always progresses.
"""

from __future__ import annotations

import time

import pytest

from app.llm.base import LLMError
from app.reflection.diagnoser import ReflectionAgent, _LLMDiagnosis
from app.reflection.models import Diagnosis, FailureCategory


class _RecordingLLM:
    """Returns a canned _LLMDiagnosis and records whether it was called."""

    def __init__(self, response: _LLMDiagnosis | None = None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.called = 0

    def complete(self, *, system: str, user: str, schema):
        self.called += 1
        if self._raises is not None:
            raise self._raises
        return self._response


class _SlowLLM:
    """Blocks longer than the diagnosis timeout to exercise the wall-clock cap."""

    def __init__(self, delay_s: float):
        self._delay_s = delay_s
        self.called = 0

    def complete(self, *, system: str, user: str, schema):
        self.called += 1
        time.sleep(self._delay_s)
        return _LLMDiagnosis(
            category=FailureCategory.UNKNOWN_ERROR, confidence=0.5, reason="late"
        )


def test_confident_heuristic_skips_the_llm():
    # The golden target-encoding case is a confident heuristic hit.
    llm = _RecordingLLM()
    agent = ReflectionAgent(llm)

    d = agent.diagnose(
        failed_capability="training",
        error="could not convert string to float: 'Yes'",
        exc_type="ValueError",
        state={},
    )

    assert d.category is FailureCategory.TARGET_ENCODING_ERROR
    assert d.source == "heuristic"
    assert llm.called == 0  # never consulted the model


def test_low_confidence_falls_back_to_llm():
    # A message no rule matches is low-confidence UNKNOWN -> consult the LLM.
    llm = _RecordingLLM(
        _LLMDiagnosis(
            category=FailureCategory.DATA_SCHEMA_ERROR,
            confidence=0.8,
            reason="model says schema",
            suggested_repair="validate_schema",
        )
    )
    agent = ReflectionAgent(llm)

    d = agent.diagnose(
        failed_capability="training",
        error="the flux capacitor destabilised",
        exc_type="RuntimeError",
        state={},
    )

    assert llm.called == 1
    assert d.category is FailureCategory.DATA_SCHEMA_ERROR
    assert d.source == "llm"
    assert d.suggested_repair == "validate_schema"


def test_llm_error_degrades_to_heuristic():
    # The provider errors -> keep the (low-confidence) heuristic, never raise.
    llm = _RecordingLLM(raises=LLMError("boom"))
    agent = ReflectionAgent(llm)

    d = agent.diagnose(
        failed_capability="training",
        error="something unrecognised",
        exc_type="RuntimeError",
        state={},
    )

    assert llm.called == 1
    assert d.category is FailureCategory.UNKNOWN_ERROR
    assert d.source == "heuristic"


def test_unexpected_llm_exception_degrades_to_heuristic():
    llm = _RecordingLLM(raises=ValueError("not an LLMError"))
    agent = ReflectionAgent(llm)

    d = agent.diagnose(
        failed_capability="training",
        error="something unrecognised",
        exc_type="RuntimeError",
        state={},
    )

    assert d.source == "heuristic"
    assert d.category is FailureCategory.UNKNOWN_ERROR


def test_llm_timeout_degrades_to_heuristic():
    # A hanging provider must not stall the graph: the wall-clock cap fires and
    # the heuristic verdict is returned.
    llm = _SlowLLM(delay_s=1.0)
    agent = ReflectionAgent(llm, diagnosis_timeout_s=0.05)

    d = agent.diagnose(
        failed_capability="training",
        error="something unrecognised",
        exc_type="RuntimeError",
        state={},
    )

    assert d.source == "heuristic"
    assert d.category is FailureCategory.UNKNOWN_ERROR


def test_diagnose_never_raises_even_with_empty_error():
    agent = ReflectionAgent(_RecordingLLM(raises=LLMError("x")))
    d = agent.diagnose(
        failed_capability="training", error="", exc_type="", state={}
    )
    assert isinstance(d, Diagnosis)
