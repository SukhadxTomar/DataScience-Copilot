"""Tests for the feature engineering tool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.tools.features import engineer_features


def test_expands_datetime_into_parts(cleaned_parquet: Path, tmp_path: Path):
    out = tmp_path / "features.parquet"
    report = engineer_features(
        cleaned_path=cleaned_parquet,
        out_path=out,
        target_column="churned",
        datetime_columns=["signup_date"],
        log_transform_columns=[],
        drop_columns=[],
    )

    df = pd.read_parquet(out)
    assert "signup_date" not in df.columns  # original dropped
    for part in ("year", "month", "day", "dayofweek"):
        assert f"signup_date_{part}" in df.columns
    assert report["expanded_datetime_columns"] == ["signup_date"]
    assert set(report["added_columns"]) == {
        f"signup_date_{p}" for p in ("year", "month", "day", "dayofweek")
    }


def test_log_transform_applied_to_positive_column(cleaned_parquet: Path, tmp_path: Path):
    original = pd.read_parquet(cleaned_parquet)["income"]
    out = tmp_path / "features.parquet"
    report = engineer_features(
        cleaned_path=cleaned_parquet,
        out_path=out,
        target_column="churned",
        datetime_columns=[],
        log_transform_columns=["income"],
        drop_columns=[],
    )

    df = pd.read_parquet(out)
    assert report["log_transformed_columns"] == ["income"]
    np.testing.assert_allclose(df["income"].to_numpy(), np.log1p(original.to_numpy()))


def test_negative_column_is_skipped_not_transformed(tmp_path: Path):
    df = pd.DataFrame({"balance": [-5.0, 10.0, 20.0], "y": [0, 1, 0]})
    cleaned = tmp_path / "cleaned.parquet"
    df.to_parquet(cleaned, index=False)

    out = tmp_path / "features.parquet"
    report = engineer_features(
        cleaned_path=cleaned,
        out_path=out,
        target_column="y",
        datetime_columns=[],
        log_transform_columns=["balance"],
        drop_columns=[],
    )

    result = pd.read_parquet(out)
    # Untouched because of the negative value...
    pd.testing.assert_series_equal(result["balance"], df["balance"])
    assert report["log_transformed_columns"] == []
    assert any("balance" in s for s in report["skipped"])


def test_target_is_never_transformed_or_dropped(cleaned_parquet: Path, tmp_path: Path):
    out = tmp_path / "features.parquet"
    engineer_features(
        cleaned_path=cleaned_parquet,
        out_path=out,
        target_column="churned",
        datetime_columns=["churned"],
        log_transform_columns=["churned"],
        drop_columns=["churned"],
    )
    df = pd.read_parquet(out)
    assert "churned" in df.columns


def test_feature_count_reported(cleaned_parquet: Path, tmp_path: Path):
    out = tmp_path / "features.parquet"
    report = engineer_features(
        cleaned_path=cleaned_parquet,
        out_path=out,
        target_column="churned",
        datetime_columns=["signup_date"],
        log_transform_columns=["income"],
        drop_columns=["plan"],
    )
    # cleaned had: signup_date, age, income, plan (4 features, target excluded).
    # signup_date -> 4 parts, drop plan: 4 - 1 (signup gone) + 4 - 1 (plan) = 6
    assert report["n_features_before"] == 4
    assert report["n_features_after"] == 6
    assert report["dropped_columns"] == ["plan"]
