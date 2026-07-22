"""Dataset profiling tool.

Produces a compact, deterministic summary of a dataset: column types,
missing values, basic statistics, and heuristic role hints. The output is
intentionally small so it can be passed to an LLM prompt directly —
raw dataframes never leave this layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

HIGH_CARDINALITY_RATIO = 0.5
ID_LIKE_RATIO = 0.98


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataset from CSV, Parquet, or Excel."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def profile_dataset(path: Path) -> dict[str, Any]:
    """Build a compact profile of the dataset at the given path."""
    df = load_dataframe(path)
    n_rows, n_cols = df.shape

    columns: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        entry: dict[str, Any] = {
            "name": str(col),
            "dtype": _semantic_dtype(series),
            "missing_pct": round(100 * n_missing / n_rows, 2) if n_rows else 0.0,
            "n_unique": n_unique,
        }

        if pd.api.types.is_numeric_dtype(series) and not _is_boolean_like(series):
            desc = series.describe()
            entry["stats"] = {
                "min": _num(desc.get("min")),
                "max": _num(desc.get("max")),
                "mean": _num(desc.get("mean")),
                "std": _num(desc.get("std")),
                "median": _num(series.median()),
            }
        else:
            top = series.value_counts(dropna=True).head(5)
            entry["top_values"] = {str(k): int(v) for k, v in top.items()}

        entry["role_hint"] = _role_hint(series, n_unique, n_rows)
        columns.append(entry)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns": columns,
        "warnings": _warnings(df, columns),
    }


def _semantic_dtype(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if _is_boolean_like(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if series.dtype == object:
        sample = series.dropna().head(50)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors="raise", format="mixed")
                return "datetime_string"
            except (ValueError, TypeError):
                pass
    return "categorical"


def _is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    uniques = set(series.dropna().unique().tolist())
    return uniques.issubset({0, 1}) and len(uniques) > 0


def _role_hint(series: pd.Series, n_unique: int, n_rows: int) -> str:
    """Classify a column as id_like, constant, binary, high_cardinality, or feature."""
    if n_rows == 0:
        return "feature"
    if n_unique <= 1:
        return "constant"
    if n_unique / n_rows >= ID_LIKE_RATIO:
        return "id_like"
    if n_unique == 2:
        return "binary"
    if series.dtype == object and n_unique / n_rows >= HIGH_CARDINALITY_RATIO:
        return "high_cardinality"
    return "feature"


def _warnings(df: pd.DataFrame, columns: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        out.append(f"{n_duplicates} duplicate rows found")
    for col in columns:
        if col["missing_pct"] > 30:
            out.append(f"Column '{col['name']}' has {col['missing_pct']}% missing values")
        if col["role_hint"] == "constant":
            out.append(f"Column '{col['name']}' is constant (no information)")
        if col["role_hint"] == "id_like":
            out.append(f"Column '{col['name']}' looks like an identifier")
    return out


def _num(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)
