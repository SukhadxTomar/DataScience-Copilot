"""Tests for the deterministic exception taxonomy (Phase A)."""

from __future__ import annotations

import pytest

from app.reflection.models import FailureCategory
from app.reflection.taxonomy import HEURISTIC_MIN_CONFIDENCE, classify


def test_xgboost_non_numeric_target_is_the_golden_case():
    # The motivating self-heal case: XGBoost/sklearn refusing a "Yes"/"No" target.
    d = classify("ValueError", "could not convert string to float: 'Yes'")

    assert d.category is FailureCategory.TARGET_ENCODING_ERROR
    assert d.confidence >= HEURISTIC_MIN_CONFIDENCE
    assert d.suggested_repair == "label_encode_target"
    assert d.source == "heuristic"
    # The offending value is extracted as evidence so the repair need not re-parse.
    assert d.evidence.get("offending_value") == "Yes"


def test_unknown_message_falls_back_to_llm_not_silent_unknown():
    # A message no rule matches must be LOW confidence UNKNOWN_ERROR — the signal
    # for the ReflectionAgent to consult the LLM rather than accept the verdict.
    d = classify("RuntimeError", "the flux capacitor destabilised unexpectedly")

    assert d.category is FailureCategory.UNKNOWN_ERROR
    assert d.confidence < HEURISTIC_MIN_CONFIDENCE
    assert d.suggested_repair is None
    assert d.source == "heuristic"


@pytest.mark.parametrize(
    "exc_type, message, expected, repair",
    [
        (
            "ValueError",
            "Input X contains NaN.",
            FailureCategory.MISSING_VALUES,
            "impute_missing_values",
        ),
        (
            "ValueError",
            "Input contains infinity or a value too large for dtype('float64').",
            FailureCategory.MISSING_VALUES,
            "remove_invalid_rows",
        ),
        (
            "ValueError",
            "could not convert string to float: hello world",  # no quotes -> feature-level
            FailureCategory.TYPE_ERROR,
            "coerce_numeric",
        ),
        (
            "TypeError",
            "'<' not supported between instances of 'str' and 'int'",
            FailureCategory.TYPE_ERROR,
            "coerce_numeric",
        ),
        (
            "ValueError",
            "X has 12 features, but RandomForestClassifier is expecting 10 features",
            FailureCategory.DATA_SCHEMA_ERROR,
            "validate_schema",
        ),
        (
            "ValueError",
            "n_splits=5 cannot be greater than the number of members in each class.",
            FailureCategory.MODEL_CONFIGURATION_ERROR,
            "reduce_cv_folds",
        ),
        (
            "MemoryError",
            "Unable to allocate 4.5 GiB for an array",
            FailureCategory.RESOURCE_LIMIT,
            None,
        ),
        (
            "LLMRateLimitError",
            "429 Too Many Requests: rate limit exceeded",
            FailureCategory.TRANSIENT_ERROR,
            "retry_llm",
        ),
        (
            "ValueError",
            "Found array with 0 sample(s) while a minimum of 1 is required.",
            FailureCategory.VALIDATION_ERROR,
            "remove_invalid_rows",
        ),
    ],
)
def test_known_failure_modes_classify_confidently(exc_type, message, expected, repair):
    d = classify(exc_type, message)
    assert d.category is expected
    assert d.confidence >= HEURISTIC_MIN_CONFIDENCE
    assert d.suggested_repair == repair


def test_planning_concerns_carry_a_planner_hint_not_a_repair():
    # RESOURCE_LIMIT is a planning concern: no artifact repair, but a hint the
    # planner can action (shrink the search space).
    d = classify("MemoryError", "Out of memory: cannot allocate memory")
    assert d.category is FailureCategory.RESOURCE_LIMIT
    assert d.suggested_repair is None
    assert d.planner_hint is not None
    assert d.planner_hint.reduce_search_space is True

    # An invalid model configuration hints a model swap.
    d2 = classify("ValueError", "Invalid parameter for estimator; unknown solver")
    assert d2.category is FailureCategory.MODEL_CONFIGURATION_ERROR
    assert d2.planner_hint is not None
    assert d2.planner_hint.requires_model_swap is True


def test_target_encoding_beats_generic_type_error_ordering():
    # The quoted-value target case must win over the broader coerce_numeric rule,
    # proving specific-before-general rule ordering holds.
    d = classify("ValueError", "could not convert string to float: 'No'")
    assert d.category is FailureCategory.TARGET_ENCODING_ERROR
    assert d.evidence.get("offending_value") == "No"


def test_classify_is_case_insensitive():
    d = classify("ValueError", "INPUT X CONTAINS NAN")
    assert d.category is FailureCategory.MISSING_VALUES
