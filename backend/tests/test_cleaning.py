"""Tests for the cleaning tool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.tools.cleaning import clean_dataset


def test_clean_drops_columns_and_writes_parquet(raw_csv: Path, tmp_path: Path):
    out = tmp_path / "cleaned.parquet"
    report = clean_dataset(
        raw_path=raw_csv,
        out_path=out,
        target_column="churned",
        drop_columns=["customer_id"],
        drop_duplicates=True,
    )

    assert out.exists()
    df = pd.read_parquet(out)
    assert "customer_id" not in df.columns
    assert "churned" in df.columns
    assert report["cleaned_path"] == str(out)
    assert report["dropped_columns"] == ["customer_id"]
    assert report["rows_after"] <= report["rows_before"]


def test_clean_never_drops_target(raw_csv: Path, tmp_path: Path):
    out = tmp_path / "cleaned.parquet"
    # Even if the plan mistakenly lists the target, it must survive.
    report = clean_dataset(
        raw_path=raw_csv,
        out_path=out,
        target_column="churned",
        drop_columns=["churned", "customer_id"],
        drop_duplicates=False,
    )
    df = pd.read_parquet(out)
    assert "churned" in df.columns
    assert "churned" not in report["dropped_columns"]


def test_clean_drops_rows_missing_target(tmp_path: Path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, None, 0]})
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, index=False)

    out = tmp_path / "cleaned.parquet"
    report = clean_dataset(
        raw_path=raw,
        out_path=out,
        target_column="y",
        drop_columns=[],
        drop_duplicates=False,
    )
    assert report["rows_after"] == 2  # the row with a missing target is gone
