"""Unit tests for the deterministic repair primitives (tools/repair_ops.py).

Each primitive is tested in isolation on a fixture parquet: it fixes the defect,
reports ``changed=True``, and is a no-op (``changed=False``) when there is
nothing to fix. The written artifact is read back and asserted on directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.tools import repair_ops


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    df.to_parquet(path, index=False)
    return path


@pytest.fixture()
def tmp_parquet(tmp_path: Path):
    def _make(df: pd.DataFrame, name: str = "data.parquet") -> Path:
        return _write_parquet(df, tmp_path / name)

    return _make


# --- label_encode_target ----------------------------------------------------


def test_label_encode_target_binary_yes_no(tmp_parquet):
    # The golden XGBoost failure case: target is "Yes"/"No".
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": ["No", "Yes", "Yes", "No"]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.label_encode_target(src, out, target_column="y")

    assert result["changed"] is True
    assert result["mapping"] == {"No": 0, "Yes": 1}
    encoded = pd.read_parquet(out)
    assert encoded["y"].tolist() == [0, 1, 1, 0]
    assert pd.api.types.is_numeric_dtype(encoded["y"])


def test_label_encode_target_multiclass_factorizes(tmp_parquet):
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["b", "a", "c"]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.label_encode_target(src, out, target_column="y")

    assert result["changed"] is True
    # sorted order -> a=0, b=1, c=2
    assert result["mapping"] == {"a": 0, "b": 1, "c": 2}
    assert pd.read_parquet(out)["y"].tolist() == [1, 0, 2]


def test_label_encode_target_already_numeric_is_noop(tmp_parquet):
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.label_encode_target(src, out, target_column="y")

    assert result["changed"] is False
    assert not out.exists()


def test_label_encode_target_missing_column_is_noop(tmp_parquet):
    src = tmp_parquet(pd.DataFrame({"x": [1, 2]}))
    out = src.parent / "out.parquet"
    result = repair_ops.label_encode_target(src, out, target_column="nope")
    assert result["changed"] is False


# --- coerce_numeric ---------------------------------------------------------


def test_coerce_numeric_converts_stringy_numbers(tmp_parquet):
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.coerce_numeric(src, out)

    assert result["changed"] is True
    assert result["coerced_columns"] == ["a"]  # b is genuinely categorical
    written = pd.read_parquet(out)
    assert pd.api.types.is_numeric_dtype(written["a"])
    assert written["b"].tolist() == ["x", "y", "z"]


def test_coerce_numeric_respects_target_and_explicit_columns(tmp_parquet):
    df = pd.DataFrame({"a": ["1", "2"], "t": ["5", "6"]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.coerce_numeric(src, out, columns=["a"], target_column="t")

    assert result["coerced_columns"] == ["a"]
    assert pd.read_parquet(out)["t"].tolist() == ["5", "6"]  # untouched


def test_coerce_numeric_noop_when_nothing_coercible(tmp_parquet):
    src = tmp_parquet(pd.DataFrame({"a": ["x", "y"]}))
    out = src.parent / "out.parquet"
    assert repair_ops.coerce_numeric(src, out)["changed"] is False


# --- impute_missing_values --------------------------------------------------


def test_impute_missing_values_median_and_mode(tmp_parquet):
    df = pd.DataFrame(
        {"num": [1.0, np.nan, 3.0], "cat": ["a", None, "a"], "t": [0, 1, 0]}
    )
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.impute_missing_values(src, out, target_column="t")

    assert result["changed"] is True
    written = pd.read_parquet(out)
    assert written["num"].isna().sum() == 0
    assert written["num"].tolist() == [1.0, 2.0, 3.0]  # median of [1,3] = 2
    assert written["cat"].tolist() == ["a", "a", "a"]


def test_impute_missing_values_skips_target(tmp_parquet):
    df = pd.DataFrame({"num": [1.0, 2.0], "t": [np.nan, 1.0]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"
    result = repair_ops.impute_missing_values(src, out, target_column="t")
    # nothing to impute in features -> no-op, target NaN left for row removal
    assert result["changed"] is False


# --- remove_invalid_rows ----------------------------------------------------


def test_remove_invalid_rows_drops_nan_target_and_inf(tmp_parquet):
    df = pd.DataFrame(
        {"x": [1.0, 2.0, np.inf, 4.0], "t": [1.0, np.nan, 0.0, 1.0]}
    )
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.remove_invalid_rows(src, out, target_column="t")

    assert result["changed"] is True
    assert result["rows_removed"] == 2  # row 1 (nan target) + row 2 (inf)
    written = pd.read_parquet(out)
    assert len(written) == 2


def test_remove_invalid_rows_noop_when_clean(tmp_parquet):
    df = pd.DataFrame({"x": [1.0, 2.0], "t": [0.0, 1.0]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"
    assert repair_ops.remove_invalid_rows(src, out, target_column="t")["changed"] is False


# --- drop_constant_columns --------------------------------------------------


def test_drop_constant_columns_drops_zero_variance(tmp_parquet):
    df = pd.DataFrame({"const": [5, 5, 5], "vary": [1, 2, 3], "t": [0, 1, 0]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.drop_constant_columns(src, out, target_column="t")

    assert result["dropped_columns"] == ["const"]
    written = pd.read_parquet(out)
    assert "const" not in written.columns
    assert "vary" in written.columns


def test_drop_constant_columns_preserves_constant_target(tmp_parquet):
    df = pd.DataFrame({"vary": [1, 2, 3], "t": [1, 1, 1]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"
    result = repair_ops.drop_constant_columns(src, out, target_column="t")
    assert result["changed"] is False  # only the target is constant, kept


# --- convert_datetime_columns -----------------------------------------------


def test_convert_datetime_columns_expands_parts(tmp_parquet):
    df = pd.DataFrame(
        {"d": ["2021-01-05", "2022-06-15", "2023-12-25"], "t": [0, 1, 0]}
    )
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.convert_datetime_columns(src, out, target_column="t")

    assert result["changed"] is True
    assert result["expanded_columns"] == ["d"]
    written = pd.read_parquet(out)
    assert "d" not in written.columns
    for part in ("year", "month", "day", "dayofweek"):
        assert f"d_{part}" in written.columns
    assert written["d_year"].tolist() == [2021.0, 2022.0, 2023.0]


def test_convert_datetime_columns_ignores_non_dates(tmp_parquet):
    src = tmp_parquet(pd.DataFrame({"s": ["a", "b", "c"], "t": [0, 1, 0]}))
    out = src.parent / "out.parquet"
    result = repair_ops.convert_datetime_columns(src, out, target_column="t")
    assert result["changed"] is False


# --- validate_schema --------------------------------------------------------


def test_validate_schema_adds_missing_drops_extra_reorders(tmp_parquet):
    df = pd.DataFrame({"b": [1, 2], "extra": [9, 9], "t": [0, 1]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"

    result = repair_ops.validate_schema(
        src, out, expected_columns=["a", "b"], target_column="t"
    )

    assert result["changed"] is True
    assert result["added_columns"] == ["a"]
    assert result["dropped_columns"] == ["extra"]
    written = pd.read_parquet(out)
    assert list(written.columns) == ["a", "b", "t"]  # expected order + target
    assert written["a"].isna().all()


def test_validate_schema_noop_when_matching(tmp_parquet):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "t": [0, 1]})
    src = tmp_parquet(df)
    out = src.parent / "out.parquet"
    result = repair_ops.validate_schema(
        src, out, expected_columns=["a", "b"], target_column="t"
    )
    assert result["changed"] is False
