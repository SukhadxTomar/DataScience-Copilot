"""Deterministic exception taxonomy.

``classify`` is a pure function: given an exception class name and message, it
returns a :class:`Diagnosis` by matching an ordered list of rules. It is the
fast, free first tier of diagnosis — no LLM, no state — and covers the known
failure modes (the XGBoost ``Yes``/``No`` target case is a pure heuristic hit).

Design contract, relied on by the ReflectionAgent (Phase C):

- A confident match returns a diagnosis with ``confidence >= HEURISTIC_MIN_CONFIDENCE``.
  The agent trusts it and skips the LLM.
- No match returns ``UNKNOWN_ERROR`` with a deliberately LOW confidence. This is a
  *routing signal*: the agent, seeing sub-threshold confidence, falls back to the
  LLM rather than treating ``UNKNOWN_ERROR`` as a final verdict. So an unfamiliar
  exception is escalated to the model, never silently swallowed.

Rules are matched in order; the first match wins, so more specific patterns are
listed before broader ones.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.reflection.models import Diagnosis, FailureCategory, PlannerHint

# The confidence floor the agent uses to decide "trust the heuristic" vs
# "ask the LLM". A no-match diagnosis sits below this on purpose.
HEURISTIC_MIN_CONFIDENCE: float = 0.6
_UNKNOWN_CONFIDENCE: float = 0.2


@dataclass(frozen=True)
class _Rule:
    """One taxonomy rule: a message regex, its resulting category/confidence,
    and optional extractors for repair evidence and a planner hint."""

    pattern: re.Pattern[str]
    category: FailureCategory
    confidence: float
    reason: str
    suggested_repair: str | None = None
    evidence_of: Callable[[re.Match[str], str], dict[str, Any]] | None = None
    hint_of: Callable[[re.Match[str], str], PlannerHint] | None = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _numeric_conversion_evidence(m: re.Match[str], _msg: str) -> dict[str, Any]:
    """Extract the offending value from a 'could not convert string to float'
    message, e.g. the ``Yes`` in ``... to float: 'Yes'``."""
    value = m.groupdict().get("value")
    return {"offending_value": value} if value else {}


# Ordered: most specific first. Each pattern is matched against the exception
# message; ``exc_type`` is folded into the message before matching so rules can
# key off either the class name or the text.
_RULES: list[_Rule] = [
    # --- TARGET_ENCODING_ERROR ---------------------------------------------
    # sklearn/xgboost refusing a non-numeric target: "could not convert string
    # to float: 'Yes'". This is the motivating self-heal case.
    _Rule(
        pattern=_rx(r"could not convert string to float:\s*'(?P<value>[^']*)'"),
        category=FailureCategory.TARGET_ENCODING_ERROR,
        confidence=0.9,
        reason="A non-numeric string reached a numeric estimator — the target "
        "or a feature column needs encoding.",
        suggested_repair="label_encode_target",
        evidence_of=_numeric_conversion_evidence,
    ),
    _Rule(
        pattern=_rx(r"unknown label type|invalid classes inferred|unknown class"),
        category=FailureCategory.TARGET_ENCODING_ERROR,
        confidence=0.85,
        reason="The target labels are not in a form the classifier accepts and "
        "must be encoded to integers.",
        suggested_repair="label_encode_target",
    ),
    # --- MISSING_VALUES -----------------------------------------------------
    _Rule(
        pattern=_rx(r"input (?:x |data )?contains nan|contains nan\b|missing values"),
        category=FailureCategory.MISSING_VALUES,
        confidence=0.85,
        reason="The data still contains NaN values where the estimator requires "
        "complete inputs.",
        suggested_repair="impute_missing_values",
    ),
    _Rule(
        pattern=_rx(r"contains infinity|infinity or a value too large"),
        category=FailureCategory.MISSING_VALUES,
        confidence=0.8,
        reason="The data contains infinite or out-of-range values that must be "
        "removed or clipped.",
        suggested_repair="remove_invalid_rows",
    ),
    # --- TYPE_ERROR ---------------------------------------------------------
    _Rule(
        pattern=_rx(r"could not convert string to float:(?!\s*')"),
        category=FailureCategory.TYPE_ERROR,
        confidence=0.7,
        reason="A column expected to be numeric holds non-numeric values and "
        "must be coerced.",
        suggested_repair="coerce_numeric",
    ),
    _Rule(
        pattern=_rx(
            r"ufunc .* not supported for the input types|"
            r"cannot perform .* with .*dtype|"
            r"'<' not supported between instances|"
            r"object of type .* has no len"
        ),
        category=FailureCategory.TYPE_ERROR,
        confidence=0.7,
        reason="A dtype mismatch prevented the operation — a column likely needs "
        "coercion or conversion.",
        suggested_repair="coerce_numeric",
    ),
    # --- DATA_SCHEMA_ERROR --------------------------------------------------
    _Rule(
        pattern=_rx(
            r"feature.* mismatch|number of features.* does not match|"
            r"columns are missing|feature names should match|"
            r"x has \d+ features, but"
        ),
        category=FailureCategory.DATA_SCHEMA_ERROR,
        confidence=0.8,
        reason="The feature schema at inference differs from training — columns "
        "must be realigned.",
        suggested_repair="validate_schema",
    ),
    _Rule(
        pattern=_rx(r"could not be parsed|error tokenizing data|unable to parse"),
        category=FailureCategory.DATA_SCHEMA_ERROR,
        confidence=0.75,
        reason="A parsing error suggests malformed rows or an unexpected file "
        "structure.",
    ),
    # --- MODEL_CONFIGURATION_ERROR (planner concern) ------------------------
    _Rule(
        pattern=_rx(
            r"n_splits=\d+ cannot be greater|"
            r"the least populated class .* has only|"
            r"minimum number of groups for any class"
        ),
        category=FailureCategory.MODEL_CONFIGURATION_ERROR,
        confidence=0.85,
        reason="Cross-validation is configured with more folds than the smallest "
        "class supports.",
        suggested_repair="reduce_cv_folds",
    ),
    _Rule(
        pattern=_rx(r"unsupported problem type|invalid parameter|unknown solver"),
        category=FailureCategory.MODEL_CONFIGURATION_ERROR,
        confidence=0.75,
        reason="The model configuration is invalid for this data — the planner "
        "should adjust the modeling strategy.",
        hint_of=lambda _m, _msg: PlannerHint(
            requires_model_swap=True,
            note="Model configuration rejected the data; try a different estimator.",
        ),
    ),
    # --- RESOURCE_LIMIT (planner concern) -----------------------------------
    _Rule(
        pattern=_rx(
            r"out of memory|unable to allocate|memoryerror|"
            r"cannot allocate memory|killed"
        ),
        category=FailureCategory.RESOURCE_LIMIT,
        confidence=0.8,
        reason="The step exhausted available memory — the planner should reduce "
        "the search space or choose a lighter model.",
        hint_of=lambda _m, _msg: PlannerHint(
            reduce_search_space=True,
            note="Ran out of memory; prefer fewer/lighter models.",
        ),
    ),
    # --- TRANSIENT_ERROR ----------------------------------------------------
    _Rule(
        pattern=_rx(
            r"rate.?limit|timed out|timeout|temporarily unavailable|"
            r"connection reset|connection aborted|503|502|429"
        ),
        category=FailureCategory.TRANSIENT_ERROR,
        confidence=0.75,
        reason="A transient provider/network condition — a bounded retry is "
        "likely to succeed.",
        suggested_repair="retry_llm",
    ),
    # --- VALIDATION_ERROR ---------------------------------------------------
    _Rule(
        pattern=_rx(
            r"found array with 0 sample|0 feature|empty data|"
            r"n_samples=0|expected a non-empty"
        ),
        category=FailureCategory.VALIDATION_ERROR,
        confidence=0.75,
        reason="The data failed a basic validity check (empty or degenerate "
        "input).",
        suggested_repair="remove_invalid_rows",
    ),
]


def classify(exc_type: str, message: str) -> Diagnosis:
    """Classify a failure from its exception type and message.

    Returns a confident :class:`Diagnosis` on a rule match, or a low-confidence
    ``UNKNOWN_ERROR`` (``confidence < HEURISTIC_MIN_CONFIDENCE``) when nothing
    matches — the signal for the ReflectionAgent to fall back to the LLM.
    """
    haystack = f"{exc_type}: {message}"

    for rule in _RULES:
        m = rule.pattern.search(haystack)
        if m is None:
            continue
        evidence = rule.evidence_of(m, message) if rule.evidence_of else {}
        hint = rule.hint_of(m, message) if rule.hint_of else None
        return Diagnosis(
            category=rule.category,
            confidence=rule.confidence,
            reason=rule.reason,
            suggested_repair=rule.suggested_repair,
            planner_hint=hint,
            evidence=evidence,
            source="heuristic",
        )

    return Diagnosis(
        category=FailureCategory.UNKNOWN_ERROR,
        confidence=_UNKNOWN_CONFIDENCE,
        reason=f"No heuristic rule matched {exc_type}. Escalating to the LLM for "
        "diagnosis.",
        source="heuristic",
    )
