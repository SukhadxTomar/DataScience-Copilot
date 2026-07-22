"""Shared test fixtures.

Builds a small, deterministic synthetic dataset that exercises every
Phase 3 tool: an id-like column, a datetime column, a right-skewed numeric
column, a normal numeric column, a categorical column, and a binary target
with real signal so the models can actually learn something.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_ROWS = 120


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.integers(18, 80, size=N_ROWS).astype(float)
    # Right-skewed, strictly positive — a natural log-transform candidate.
    income = rng.lognormal(mean=10.5, sigma=0.6, size=N_ROWS).round(2)
    plan = rng.choice(["basic", "plus", "pro"], size=N_ROWS)
    signup_date = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 900, size=N_ROWS), unit="D"
    )

    # Target with genuine signal: older, lower-income customers churn more.
    logit = 0.04 * (55 - age) + 1.2 * (income < 30000) - 0.5
    prob = 1 / (1 + np.exp(-logit))
    churned = (rng.random(N_ROWS) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1000 + N_ROWS),  # id-like
            "signup_date": signup_date.strftime("%Y-%m-%d"),  # datetime string
            "age": age,
            "income": income,
            "plan": plan,
            "churned": churned,
        }
    )


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    """A raw CSV on disk, as if freshly uploaded."""
    path = tmp_path / "raw.csv"
    make_dataframe().to_csv(path, index=False)
    return path


@pytest.fixture
def cleaned_parquet(tmp_path: Path) -> Path:
    """A cleaned Parquet with the id column already dropped, as the cleaning
    tool would produce it."""
    path = tmp_path / "cleaned.parquet"
    make_dataframe().drop(columns=["customer_id"]).to_parquet(path, index=False)
    return path
