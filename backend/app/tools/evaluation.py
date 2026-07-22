"""Model evaluation tool.

Evaluates the trained pipeline on a held-out test split and returns
metrics suitable for both technical review and business reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_model(
    data_path: Path,
    model_path: Path,
    target_column: str,
    problem_type: str,
) -> dict[str, Any]:
    """Retrain on a train split and score on the held-out test split.

    The persisted model was fitted on all data (best for deployment), so
    for honest metrics we clone-and-refit on the train split only.
    """
    df = pd.read_parquet(data_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])

    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    from sklearn.base import clone

    pipeline = clone(joblib.load(model_path))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
    else:
        metrics = {
            "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }

    return {
        "metrics": metrics,
        "n_test_samples": int(len(y_test)),
        "test_size": TEST_SIZE,
    }
