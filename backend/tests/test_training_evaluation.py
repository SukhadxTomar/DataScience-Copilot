"""Tests for the training and evaluation tools.

These run real sklearn/XGBoost/LightGBM pipelines on the small synthetic
dataset, so they double as a smoke test that the ML dependencies are
installed and the leak-safe Pipeline wiring holds together.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app.tools.evaluation import evaluate_model
from app.tools.training import train_models


def _features(cleaned_parquet: Path, tmp_path: Path) -> Path:
    """A ready-to-model Parquet (drop the id-free cleaned data straight in)."""
    path = tmp_path / "features.parquet"
    pd.read_parquet(cleaned_parquet).to_parquet(path, index=False)
    return path


def test_training_produces_leaderboard_and_model(cleaned_parquet: Path, tmp_path: Path):
    data = _features(cleaned_parquet, tmp_path)
    artifacts = tmp_path / "artifacts"

    report = train_models(
        data_path=data,
        target_column="churned",
        problem_type="classification",
        artifacts_dir=artifacts,
    )

    assert Path(report["model_path"]).exists()
    assert report["metric"] == "roc_auc"
    # Every candidate classifier should appear in the leaderboard.
    models = {r["model"] for r in report["results"]}
    assert {"logistic_regression", "random_forest", "xgboost", "lightgbm"} <= models
    # Leaderboard is sorted best-first.
    means = [r["cv_mean"] for r in report["results"]]
    assert means == sorted(means, reverse=True)
    assert report["best_model"] in models

    # The persisted object is a fitted sklearn pipeline that can predict.
    pipeline = joblib.load(report["model_path"])
    X = pd.read_parquet(data).drop(columns=["churned"])
    assert len(pipeline.predict(X)) == len(X)


def test_evaluation_returns_classification_metrics(cleaned_parquet: Path, tmp_path: Path):
    data = _features(cleaned_parquet, tmp_path)
    artifacts = tmp_path / "artifacts"

    train = train_models(
        data_path=data,
        target_column="churned",
        problem_type="classification",
        artifacts_dir=artifacts,
    )
    report = evaluate_model(
        data_path=data,
        model_path=Path(train["model_path"]),
        target_column="churned",
        problem_type="classification",
    )

    metrics = report["metrics"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
    assert report["n_test_samples"] > 0


def test_regression_path(tmp_path: Path):
    df = pd.DataFrame(
        {
            "x1": list(range(100)),
            "x2": [i * 0.5 for i in range(100)],
            "y": [3 * i + 2 for i in range(100)],  # perfectly linear
        }
    )
    data = tmp_path / "features.parquet"
    df.to_parquet(data, index=False)

    train = train_models(
        data_path=data,
        target_column="y",
        problem_type="regression",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert train["metric"] == "neg_root_mean_squared_error"
    assert "ridge" in {r["model"] for r in train["results"]}

    report = evaluate_model(
        data_path=data,
        model_path=Path(train["model_path"]),
        target_column="y",
        problem_type="regression",
    )
    assert set(report["metrics"]) == {"rmse", "mae", "r2"}
