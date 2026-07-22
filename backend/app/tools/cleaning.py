"""Data cleaning tool.

Executes a cleaning plan produced by the cleaning agent. The plan is
plain data (which columns to drop, whether to drop duplicates); this
module does the actual dataframe work and reports what changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.profiler import load_dataframe


def clean_dataset(
    raw_path: Path,
    out_path: Path,
    target_column: str,
    drop_columns: list[str],
    drop_duplicates: bool,
) -> dict[str, Any]:
    """Apply the cleaning plan and write the result as Parquet."""
    df = load_dataframe(raw_path)
    rows_before = len(df)

    dropped_columns = [c for c in drop_columns if c in df.columns and c != target_column]
    df = df.drop(columns=dropped_columns)

    if drop_duplicates:
        df = df.drop_duplicates()

    # Rows without a target value cannot be used for supervised learning.
    df = df.dropna(subset=[target_column])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "rows_before": rows_before,
        "rows_after": len(df),
        "rows_removed": rows_before - len(df),
        "dropped_columns": dropped_columns,
        "cleaned_path": str(out_path),
    }
