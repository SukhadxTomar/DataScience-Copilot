"""Unit tests for the repair registry and its runners.

Covers the registry contract (registration, category lookup, and the crucial
invariant that no planning action is registered as a repair) and each runner's
adaptation of state + diagnosis to a RepairResult.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Importing repairs populates REPAIR_REGISTRY as a side effect.
from app.reflection import repairs  # noqa: F401
from app.reflection.models import Diagnosis, FailureCategory
from app.reflection.repair_registry import (
    REPAIR_REGISTRY,
    candidates_for,
    get,
)


def _diag(category: FailureCategory, **evidence) -> Diagnosis:
    return Diagnosis(
        category=category,
        confidence=0.9,
        reason="test",
        evidence=evidence,
    )


def _state_with_features(path: Path, target: str = "y") -> dict:
    return {
        "problem_spec": {"target_column": target},
        "feature_report": {"features_path": str(path)},
    }


# --- registry contract ------------------------------------------------------


def test_registry_is_populated():
    assert REPAIR_REGISTRY, "importing repairs must register capabilities"
    assert "label_encode_target" in REPAIR_REGISTRY


def test_switch_model_is_not_a_repair():
    # The architectural invariant: model swap is a planner concern, never a repair.
    assert "switch_model" not in REPAIR_REGISTRY


def test_candidates_for_returns_registration_order():
    cands = candidates_for(FailureCategory.TARGET_ENCODING_ERROR)
    assert [c.name for c in cands] == ["label_encode_target"]


def test_candidates_for_multi_handler_category():
    names = {c.name for c in candidates_for(FailureCategory.DATA_SCHEMA_ERROR)}
    assert {"convert_datetime_columns", "drop_constant_columns", "validate_schema"} <= names


def test_every_capability_handles_at_least_one_category():
    for cap in REPAIR_REGISTRY.values():
        assert cap.handles, f"{cap.name} handles no category"


def test_get_unknown_returns_none():
    assert get("does_not_exist") is None


# --- runner: label_encode_target (golden path) ------------------------------


def test_label_encode_runner_rewrites_features_in_place(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"x": [1.0, 2.0], "y": ["No", "Yes"]}).to_parquet(path, index=False)
    state = _state_with_features(path)

    result = get("label_encode_target").runner(
        state, _diag(FailureCategory.TARGET_ENCODING_ERROR)
    )

    assert result.applied is True
    assert result.changed is True
    assert result.artifact_key == "feature_report"
    assert result.new_artifact_path == str(path)  # rewritten in place
    assert pd.read_parquet(path)["y"].tolist() == [0, 1]


def test_runner_reports_missing_artifact_without_raising(tmp_path):
    state = {"problem_spec": {"target_column": "y"}}  # no feature_report
    result = get("label_encode_target").runner(
        state, _diag(FailureCategory.TARGET_ENCODING_ERROR)
    )
    assert result.applied is False
    assert result.changed is False
    assert result.error == "artifact_not_found"


def test_coerce_numeric_runner_uses_evidence_columns(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"], "y": [0, 1]}).to_parquet(
        path, index=False
    )
    state = _state_with_features(path)

    result = get("coerce_numeric").runner(
        state, _diag(FailureCategory.TYPE_ERROR, columns=["a"])
    )

    assert result.changed is True
    assert pd.api.types.is_numeric_dtype(pd.read_parquet(path)["a"])


# --- runner: validate_schema needs expected columns from training ----------


def test_validate_schema_runner_without_expected_is_not_applied(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_parquet(path, index=False)
    state = _state_with_features(path)  # no training_report

    result = get("validate_schema").runner(
        state, _diag(FailureCategory.DATA_SCHEMA_ERROR)
    )

    assert result.applied is False
    assert result.error == "no_expected_schema"


def test_validate_schema_runner_aligns_to_expected(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"b": [1, 2], "extra": [9, 9], "y": [0, 1]}).to_parquet(
        path, index=False
    )
    state = _state_with_features(path)
    state["training_report"] = {"feature_columns": ["a", "b"]}

    result = get("validate_schema").runner(
        state, _diag(FailureCategory.DATA_SCHEMA_ERROR)
    )

    assert result.changed is True
    assert list(pd.read_parquet(path).columns) == ["a", "b", "y"]


# --- knob / signal runners --------------------------------------------------


def test_reduce_cv_folds_runner_proposes_lower_fold_via_knob():
    state = {"problem_spec": {"target_column": "y", "cv_folds": 5}}
    result = get("reduce_cv_folds").runner(
        state, _diag(FailureCategory.MODEL_CONFIGURATION_ERROR)
    )
    assert result.applied is True
    assert result.changed is True
    assert result.artifact_key == "cv_folds"  # a spec knob, not a dataset
    assert result.new_artifact_path is None
    assert "4" in result.detail


def test_reduce_cv_folds_runner_at_floor_is_noop():
    state = {"problem_spec": {"target_column": "y", "cv_folds": 2}}
    result = get("reduce_cv_folds").runner(
        state, _diag(FailureCategory.MODEL_CONFIGURATION_ERROR)
    )
    assert result.applied is True
    assert result.changed is False


def test_retry_llm_runner_signals_retry_without_artifact_change():
    result = get("retry_llm").runner({}, _diag(FailureCategory.TRANSIENT_ERROR))
    assert result.applied is True
    assert result.changed is True
    assert result.new_artifact_path is None
    assert result.artifact_key is None
