"""Feature engineering tool.

Executes a feature plan produced by the feature agent. The plan is plain
data (which datetime columns to expand, which skewed columns to
log-transform, which extra columns to drop); this module does the actual
dataframe work and reports what changed.

Every transform here is **row-wise** — it maps each row independently and
learns nothing from the column as a whole. That is deliberate: parameter-
learning steps (scaling, imputation, encoding) live inside the training
Pipeline so they are fitted on training folds only. Doing them here, before
the train/test split, would leak information from the test set. So this
layer is safe to run on the full dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Datetime parts extracted from each expanded datetime column.
_DATETIME_PARTS = ("year", "month", "day", "dayofweek")


def engineer_features(
    cleaned_path: Path,
    out_path: Path,
    target_column: str,
    datetime_columns: list[str],
    log_transform_columns: list[str],
    drop_columns: list[str],
) -> dict[str, Any]:
    """Apply the feature plan and write the result as Parquet.

    Returns a report describing exactly what changed, so the plan the agent
    proposed can be reconciled against what was actually possible on the
    real data.
    """
    df = pd.read_parquet(cleaned_path)
    n_features_before = df.shape[1] - 1  # exclude the target

    added_columns: list[str] = []
    expanded_columns: list[str] = []
    transformed_columns: list[str] = []
    skipped: list[str] = []

    # 1. Expand datetime columns into numeric parts, then drop the original.
    for col in datetime_columns:
        if col not in df.columns or col == target_column:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() == 0:
            skipped.append(f"{col} (not parseable as datetime)")
            continue
        for part in _DATETIME_PARTS:
            new_col = f"{col}_{part}"
            df[new_col] = getattr(parsed.dt, part).astype("float64")
            added_columns.append(new_col)
        df = df.drop(columns=[col])
        expanded_columns.append(col)

    # 2. Log-transform skewed, non-negative numeric columns (log1p is safe at 0).
    for col in log_transform_columns:
        if col not in df.columns or col == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            skipped.append(f"{col} (not numeric, cannot log-transform)")
            continue
        if float(df[col].min(skipna=True)) < 0:
            skipped.append(f"{col} (has negatives, log-transform unsafe)")
            continue
        df[col] = np.log1p(df[col])
        transformed_columns.append(col)

    # 3. Drop any remaining columns the plan flagged (never the target).
    dropped_columns = [
        c for c in drop_columns if c in df.columns and c != target_column
    ]
    df = df.drop(columns=dropped_columns)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "features_path": str(out_path),
        "n_features_before": n_features_before,
        "n_features_after": df.shape[1] - 1,
        "expanded_datetime_columns": expanded_columns,
        "added_columns": added_columns,
        "log_transformed_columns": transformed_columns,
        "dropped_columns": dropped_columns,
        "skipped": skipped,
    }
