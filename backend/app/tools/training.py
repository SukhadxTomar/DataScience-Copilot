"""Model training tool.

Trains several classifiers or regressors on a prepared dataset and
returns their cross-validated scores. Preprocessing (imputation, scaling,
encoding) is part of each model's sklearn Pipeline, so it is fitted only
on training folds — no leakage into validation data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42
CV_FOLDS = 5


def train_models(
    data_path: Path,
    target_column: str,
    problem_type: str,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Train candidate models and persist the best one.

    Returns per-model CV scores and the path of the winning model.
    """
    df = pd.read_parquet(data_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    candidates = _candidates(problem_type)
    scoring = "roc_auc" if problem_type == "classification" else "neg_root_mean_squared_error"

    results: list[dict[str, Any]] = []
    best_name, best_score, best_pipeline = None, float("-inf"), None

    for name, estimator in candidates.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
        scores = cross_val_score(pipeline, X, y, cv=CV_FOLDS, scoring=scoring, n_jobs=-1)
        mean_score = float(scores.mean())
        results.append({
            "model": name,
            "metric": scoring,
            "cv_mean": round(mean_score, 4),
            "cv_std": round(float(scores.std()), 4),
        })
        if mean_score > best_score:
            best_name, best_score, best_pipeline = name, mean_score, pipeline

    # Refit the winner on all data and persist it.
    best_pipeline.fit(X, y)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    joblib.dump(best_pipeline, model_path)

    return {
        "results": sorted(results, key=lambda r: r["cv_mean"], reverse=True),
        "best_model": best_name,
        "best_score": round(best_score, 4),
        "metric": scoring,
        "model_path": str(model_path),
        "feature_columns": X.columns.tolist(),
        "n_samples": len(df),
    }


def _candidates(problem_type: str) -> dict[str, Any]:
    if problem_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            "xgboost": XGBClassifier(
                n_estimators=200, random_state=RANDOM_STATE,
                eval_metric="logloss", verbosity=0,
            ),
            "lightgbm": LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1),
        }
    if problem_type == "regression":
        return {
            "ridge": Ridge(random_state=RANDOM_STATE),
            "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
            "xgboost": XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0),
            "lightgbm": LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE, verbose=-1),
        }
    raise ValueError(f"Unsupported problem type: {problem_type}")
