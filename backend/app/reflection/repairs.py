"""Repair capability runners and registrations.

Each runner adapts the fixed ``(state, diagnosis) -> RepairResult`` contract to a
pure primitive in ``app/tools/repair_ops.py``: it resolves the target artifact
path from state, reads any evidence the diagnosis extracted, calls the primitive,
and packages the outcome as a :class:`RepairResult`. Importing this module
populates ``REPAIR_REGISTRY``.

Design notes:

- Most repairs target the **features artifact** (``feature_report.features_path``)
  — the last data artifact before modeling, where encoding/NaN/dtype/schema
  failures surface. They rewrite it **in place**, so the retried capability
  re-reads the corrected data with no path change.
- ``reduce_cv_folds`` is the one repair that touches a spec/state *knob* rather
  than a dataset: it records a lower fold count for the training tool to honour.
  It changes no source. (Wiring the tool to read the knob lands in Phase E.)
- ``retry_llm`` changes no artifact at all — it signals a plain bounded re-run
  for transient failures. ``changed=True`` here means "a retry is warranted",
  keeping the node's applied+changed gate meaningful.
- Runners never raise for expected conditions (missing artifact, nothing to fix);
  they return ``applied=False`` / ``changed=False`` so the node owns the budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.graph.state import RunState
from app.reflection.models import Diagnosis, FailureCategory, RepairResult
from app.reflection.repair_registry import RepairCapability, register
from app.tools import repair_ops

# The state key whose artifact the data repairs operate on, and where the
# feature dataset path lives inside it.
_FEATURES_KEY = "feature_report"
_FEATURES_PATH_FIELD = "features_path"


def _features_path(state: RunState) -> Path | None:
    report = state.get(_FEATURES_KEY)
    if not isinstance(report, dict):
        return None
    path = report.get(_FEATURES_PATH_FIELD)
    return Path(path) if path else None


def _target_column(state: RunState) -> str | None:
    spec = state.get("problem_spec")
    if isinstance(spec, dict):
        return spec.get("target_column")
    return None


def _missing_artifact(repair_name: str) -> RepairResult:
    return RepairResult(
        repair_name=repair_name,
        applied=False,
        changed=False,
        detail=f"{_FEATURES_KEY}.{_FEATURES_PATH_FIELD} not available in state",
        error="artifact_not_found",
    )


def _from_op(
    repair_name: str, artifact_key: str, in_path: Path, op_result: dict[str, Any]
) -> RepairResult:
    """Build a RepairResult from a repair_ops return dict. The primitives write
    in place (out_path == in_path), so a change is reflected at the same path."""
    changed = bool(op_result.get("changed"))
    return RepairResult(
        repair_name=repair_name,
        applied=True,
        changed=changed,
        detail=op_result.get("detail", ""),
        artifact_key=artifact_key if changed else None,
        new_artifact_path=str(in_path) if changed else None,
    )


# --- artifact repairs (rewrite the features dataset in place) ---------------


def _run_label_encode_target(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    path = _features_path(state)
    target = _target_column(state)
    if path is None or target is None:
        return _missing_artifact("label_encode_target")
    result = repair_ops.label_encode_target(path, path, target_column=target)
    return _from_op("label_encode_target", _FEATURES_KEY, path, result)


def _run_coerce_numeric(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    path = _features_path(state)
    if path is None:
        return _missing_artifact("coerce_numeric")
    # If the diagnosis pinned a specific column, target just that one.
    cols = diagnosis.evidence.get("columns")
    result = repair_ops.coerce_numeric(
        path, path, columns=cols, target_column=_target_column(state)
    )
    return _from_op("coerce_numeric", _FEATURES_KEY, path, result)


def _run_impute_missing_values(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    path = _features_path(state)
    if path is None:
        return _missing_artifact("impute_missing_values")
    result = repair_ops.impute_missing_values(
        path, path, target_column=_target_column(state)
    )
    return _from_op("impute_missing_values", _FEATURES_KEY, path, result)


def _run_remove_invalid_rows(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    path = _features_path(state)
    if path is None:
        return _missing_artifact("remove_invalid_rows")
    result = repair_ops.remove_invalid_rows(
        path, path, target_column=_target_column(state)
    )
    return _from_op("remove_invalid_rows", _FEATURES_KEY, path, result)


def _run_drop_constant_columns(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    path = _features_path(state)
    if path is None:
        return _missing_artifact("drop_constant_columns")
    result = repair_ops.drop_constant_columns(
        path, path, target_column=_target_column(state)
    )
    return _from_op("drop_constant_columns", _FEATURES_KEY, path, result)


def _run_convert_datetime_columns(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    path = _features_path(state)
    if path is None:
        return _missing_artifact("convert_datetime_columns")
    cols = diagnosis.evidence.get("columns")
    result = repair_ops.convert_datetime_columns(
        path, path, columns=cols, target_column=_target_column(state)
    )
    return _from_op("convert_datetime_columns", _FEATURES_KEY, path, result)


def _run_validate_schema(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    path = _features_path(state)
    if path is None:
        return _missing_artifact("validate_schema")
    # The expected feature set comes from the last successful training report,
    # which records feature_columns; without it there is nothing to align to.
    training = state.get("training_report")
    expected = training.get("feature_columns") if isinstance(training, dict) else None
    if not expected:
        return RepairResult(
            repair_name="validate_schema",
            applied=False,
            changed=False,
            detail="no expected feature_columns available to align to",
            error="no_expected_schema",
        )
    result = repair_ops.validate_schema(
        path, path, expected_columns=expected, target_column=_target_column(state)
    )
    return _from_op("validate_schema", _FEATURES_KEY, path, result)


# --- knob / signal repairs (no dataset rewrite) ----------------------------


def _run_reduce_cv_folds(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    """Lower the CV fold count the training tool will use.

    This does not rewrite a dataset; it records a spec/state knob. The value is
    surfaced via ``new_artifact_path=None`` and a dedicated detail; the reflect
    node persists the knob into state (Phase E), and the training tool honours
    it. Never edits source.
    """
    current = _current_cv_folds(state)
    proposed = max(2, current - 1)
    if proposed >= current:
        return RepairResult(
            repair_name="reduce_cv_folds",
            applied=True,
            changed=False,
            detail=f"cv_folds already at floor ({current})",
        )
    return RepairResult(
        repair_name="reduce_cv_folds",
        applied=True,
        changed=True,
        detail=f"reduce cv_folds {current} -> {proposed}",
        artifact_key="cv_folds",
    )


def _current_cv_folds(state: RunState) -> int:
    spec = state.get("problem_spec")
    if isinstance(spec, dict) and isinstance(spec.get("cv_folds"), int):
        return spec["cv_folds"]
    # Mirror the training tool's default when no override has been set yet.
    return 5


def _run_retry_llm(state: RunState, diagnosis: Diagnosis) -> RepairResult:
    """Signal a plain bounded re-run for a transient failure. No artifact change;
    ``changed=True`` means 'a retry is warranted'."""
    return RepairResult(
        repair_name="retry_llm",
        applied=True,
        changed=True,
        detail="transient failure — retry the capability",
    )


# --- registrations (order matters for candidates_for fallback priority) -----

_CAPABILITIES: list[RepairCapability] = [
    RepairCapability(
        name="label_encode_target",
        handles=frozenset({FailureCategory.TARGET_ENCODING_ERROR}),
        runner=_run_label_encode_target,
        description="Encode a non-numeric target column to integers.",
    ),
    RepairCapability(
        name="coerce_numeric",
        handles=frozenset({FailureCategory.TYPE_ERROR}),
        runner=_run_coerce_numeric,
        description="Coerce numeric-in-disguise object columns to numbers.",
    ),
    RepairCapability(
        name="convert_datetime_columns",
        handles=frozenset(
            {FailureCategory.TYPE_ERROR, FailureCategory.DATA_SCHEMA_ERROR}
        ),
        runner=_run_convert_datetime_columns,
        description="Expand object date columns into numeric parts.",
    ),
    RepairCapability(
        name="impute_missing_values",
        handles=frozenset({FailureCategory.MISSING_VALUES}),
        runner=_run_impute_missing_values,
        description="Fill NaNs (median / most-frequent).",
    ),
    RepairCapability(
        name="remove_invalid_rows",
        handles=frozenset(
            {FailureCategory.MISSING_VALUES, FailureCategory.VALIDATION_ERROR}
        ),
        runner=_run_remove_invalid_rows,
        description="Drop rows with missing target or infinite values.",
    ),
    RepairCapability(
        name="drop_constant_columns",
        handles=frozenset(
            {FailureCategory.DATA_SCHEMA_ERROR, FailureCategory.MODEL_CONFIGURATION_ERROR}
        ),
        runner=_run_drop_constant_columns,
        description="Drop zero-variance / all-NaN columns.",
    ),
    RepairCapability(
        name="validate_schema",
        handles=frozenset({FailureCategory.DATA_SCHEMA_ERROR}),
        runner=_run_validate_schema,
        description="Realign feature columns to the model's expected set.",
    ),
    RepairCapability(
        name="reduce_cv_folds",
        handles=frozenset({FailureCategory.MODEL_CONFIGURATION_ERROR}),
        runner=_run_reduce_cv_folds,
        description="Lower the CV fold count when a class has too few samples.",
    ),
    RepairCapability(
        name="retry_llm",
        handles=frozenset({FailureCategory.TRANSIENT_ERROR}),
        runner=_run_retry_llm,
        description="Signal a bounded plain re-run for transient failures.",
    ),
]

for _cap in _CAPABILITIES:
    register(_cap)
