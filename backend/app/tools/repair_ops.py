"""Deterministic artifact-repair primitives.

Each function here fixes one class of data defect in a dataset on disk and
writes the corrected result to ``out_path``. They mirror the shape of the other
tools (``tools/cleaning.py``, ``tools/features.py``): pure dataframe work with a
``(in_path, out_path, ...) -> dict`` signature, no state, no LLM, no LangGraph.
That keeps them trivially unit-testable and reusable.

These primitives are the mechanics behind the repair capabilities in
``app/reflection/repairs.py``. Correctness is proven here in isolation; the
repair layer only wires diagnosis evidence to the right primitive.

Every function reports what it changed. A ``changed`` flag lets the caller tell
"the repair ran but the artifact is identical" (so there is no point retrying)
from "the repair fixed something" — the reflection node uses this to avoid
burning a retry on a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Common textual spellings of binary labels, mapped to 0/1. Checked
# case-insensitively. Ordering within a pair is (zero_label, one_label).
_BINARY_LABEL_MAPS: tuple[tuple[str, str], ...] = (
    ("no", "yes"),
    ("false", "true"),
    ("n", "y"),
    ("negative", "positive"),
    ("absent", "present"),
    ("0", "1"),
)


def _read(in_path: Path) -> pd.DataFrame:
    return pd.read_parquet(in_path)


def _write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def label_encode_target(
    in_path: Path,
    out_path: Path,
    target_column: str,
) -> dict[str, Any]:
    """Encode a non-numeric target column to integers.

    Binary targets with recognisable labels (Yes/No, True/False, ...) map to
    0/1 with the "positive" label as 1. Any other non-numeric target is
    factorized to a stable 0..k-1 integer code. An already-numeric target is
    left untouched (``changed=False``).

    Returns the mapping so it can be persisted (e.g. into ``problem_spec``) for
    later decoding of predictions.
    """
    df = _read(in_path)
    if target_column not in df.columns:
        return {
            "changed": False,
            "detail": f"target '{target_column}' not in columns",
            "mapping": {},
            "out_path": None,
        }

    series = df[target_column]
    if pd.api.types.is_numeric_dtype(series):
        return {
            "changed": False,
            "detail": f"target '{target_column}' already numeric",
            "mapping": {},
            "out_path": None,
        }

    mapping = _encode_series(series)
    df[target_column] = series.map(mapping)
    _write(df, out_path)
    return {
        "changed": True,
        "detail": f"encoded target '{target_column}' -> {mapping}",
        "mapping": mapping,
        "out_path": str(out_path),
    }


def _encode_series(series: pd.Series) -> dict[Any, int]:
    """Build a value->int mapping, preferring a known binary convention."""
    values = series.dropna().unique().tolist()
    lowered = {str(v).strip().lower(): v for v in values}

    if len(lowered) == 2:
        for zero_label, one_label in _BINARY_LABEL_MAPS:
            if set(lowered) == {zero_label, one_label}:
                return {lowered[zero_label]: 0, lowered[one_label]: 1}

    # Fallback: stable factorization in sorted order for determinism.
    ordered = sorted(values, key=lambda v: str(v))
    return {v: i for i, v in enumerate(ordered)}


def coerce_numeric(
    in_path: Path,
    out_path: Path,
    columns: list[str] | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Coerce object/string columns that are numeric-in-disguise to numbers.

    Non-parseable entries become NaN (``pd.to_numeric(errors="coerce")``), to be
    imputed downstream. When ``columns`` is None, every non-numeric column
    (except the target) is attempted; a column is only rewritten if coercion
    yields at least one real number, so genuinely categorical columns are left
    alone.
    """
    df = _read(in_path)
    candidates = columns if columns is not None else [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]

    coerced: list[str] = []
    for col in candidates:
        if col not in df.columns or col == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only accept the coercion if it actually produced numbers; otherwise the
        # column is really categorical and we would just destroy it.
        if converted.notna().any():
            df[col] = converted
            coerced.append(col)

    if not coerced:
        return {"changed": False, "detail": "no columns were coercible to numeric",
                "coerced_columns": [], "out_path": None}
    _write(df, out_path)
    return {
        "changed": True,
        "detail": f"coerced to numeric: {coerced}",
        "coerced_columns": coerced,
        "out_path": str(out_path),
    }


def impute_missing_values(
    in_path: Path,
    out_path: Path,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Fill NaNs: median for numeric columns, most-frequent for the rest.

    The target column is never imputed — rows with a missing target are not
    learnable and are handled by ``remove_invalid_rows`` instead.
    """
    df = _read(in_path)
    filled: dict[str, str] = {}

    for col in df.columns:
        if col == target_column:
            continue
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median()
            strategy = "median"
        else:
            modes = df[col].mode(dropna=True)
            if modes.empty:
                continue
            value = modes.iloc[0]
            strategy = "most_frequent"
        if pd.isna(value):
            continue
        df[col] = df[col].fillna(value)
        filled[col] = strategy

    if not filled:
        return {"changed": False, "detail": "no missing values to impute",
                "imputed_columns": {}, "out_path": None}
    _write(df, out_path)
    return {
        "changed": True,
        "detail": f"imputed {list(filled)}",
        "imputed_columns": filled,
        "out_path": str(out_path),
    }


def remove_invalid_rows(
    in_path: Path,
    out_path: Path,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Drop rows that cannot be used for learning.

    Removes rows with a missing target (if ``target_column`` given) and rows
    containing +/-inf in any numeric column. NaNs in features are left for
    imputation; only the unrecoverable rows go.
    """
    df = _read(in_path)
    rows_before = len(df)

    if target_column and target_column in df.columns:
        df = df.dropna(subset=[target_column])

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        finite_mask = np.isfinite(numeric).all(axis=1)
        df = df[finite_mask]

    removed = rows_before - len(df)
    if removed == 0:
        return {"changed": False, "detail": "no invalid rows found",
                "rows_removed": 0, "out_path": None}
    _write(df, out_path)
    return {
        "changed": True,
        "detail": f"removed {removed} invalid row(s)",
        "rows_removed": removed,
        "rows_before": rows_before,
        "rows_after": len(df),
        "out_path": str(out_path),
    }


def drop_constant_columns(
    in_path: Path,
    out_path: Path,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Drop zero-variance and all-NaN columns.

    A column with a single distinct value (or none) carries no signal and can
    break some estimators; dropping it is always safe. The target is preserved
    even if constant, so the failure surfaces honestly rather than silently
    losing the label.
    """
    df = _read(in_path)
    dropped: list[str] = []
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].nunique(dropna=True) <= 1:
            dropped.append(col)

    if not dropped:
        return {"changed": False, "detail": "no constant columns found",
                "dropped_columns": [], "out_path": None}
    df = df.drop(columns=dropped)
    _write(df, out_path)
    return {
        "changed": True,
        "detail": f"dropped constant columns {dropped}",
        "dropped_columns": dropped,
        "out_path": str(out_path),
    }


def convert_datetime_columns(
    in_path: Path,
    out_path: Path,
    columns: list[str] | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Expand object columns that parse as dates into numeric parts.

    Mirrors the datetime expansion in ``tools/features.py`` (year/month/day/
    dayofweek) so a datetime that slipped through gets the same treatment.
    When ``columns`` is None, every object column is probed and only those that
    parse cleanly enough (majority non-null) are expanded; the original is
    dropped.
    """
    df = _read(in_path)
    parts = ("year", "month", "day", "dayofweek")

    candidates = columns if columns is not None else [
        c for c in df.columns
        if c != target_column and not pd.api.types.is_numeric_dtype(df[c])
    ]

    expanded: list[str] = []
    added: list[str] = []
    for col in candidates:
        if col not in df.columns or col == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        # Require a majority to parse, else it's not really a date column.
        if parsed.notna().mean() < 0.5:
            continue
        for part in parts:
            new_col = f"{col}_{part}"
            df[new_col] = getattr(parsed.dt, part).astype("float64")
            added.append(new_col)
        df = df.drop(columns=[col])
        expanded.append(col)

    if not expanded:
        return {"changed": False, "detail": "no datetime columns to convert",
                "expanded_columns": [], "added_columns": [], "out_path": None}
    _write(df, out_path)
    return {
        "changed": True,
        "detail": f"expanded datetime columns {expanded}",
        "expanded_columns": expanded,
        "added_columns": added,
        "out_path": str(out_path),
    }


def validate_schema(
    in_path: Path,
    out_path: Path,
    expected_columns: list[str],
    target_column: str | None = None,
) -> dict[str, Any]:
    """Realign a dataset's feature columns to an expected set.

    Adds any missing expected columns as all-NaN (to be imputed) and drops
    unexpected extras, then reorders to match ``expected_columns``. The target
    column is always kept regardless of the expected set. This fixes the
    "X has N features, but estimator expects M" mismatch.
    """
    df = _read(in_path)
    expected = list(expected_columns)

    keep_target = target_column and target_column in df.columns
    current_features = [c for c in df.columns if c != target_column]

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in current_features if c not in expected]

    for col in missing:
        df[col] = np.nan
    if extra:
        df = df.drop(columns=extra)

    ordered = expected + ([target_column] if keep_target else [])
    df = df[[c for c in ordered if c in df.columns]]

    if not missing and not extra:
        return {"changed": False, "detail": "schema already matches expected",
                "added_columns": [], "dropped_columns": [], "out_path": None}
    _write(df, out_path)
    return {
        "changed": True,
        "detail": f"realigned schema (+{missing}, -{extra})",
        "added_columns": missing,
        "dropped_columns": extra,
        "out_path": str(out_path),
    }
