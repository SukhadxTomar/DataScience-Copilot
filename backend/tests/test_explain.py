"""Tests for the SHAP explainability tool.

Builds tiny fitted pipelines (same preprocessor shape as training.py) and
exercises the tree-SHAP, linear-SHAP, permutation-fallback, and
last-resort-unavailable paths — plus determinism.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import app.tools.explain as explain_mod
from app.tools.explain import explain_model
from tests.conftest import make_dataframe

pytestmark = pytest.mark.filterwarnings("ignore")


def _build_pipeline(tmp_path: Path, estimator) -> tuple[Path, Path]:
    """Fit `estimator` inside the training-style preprocessor; return (data, model)."""
    df = make_dataframe().drop(columns=["customer_id", "signup_date"])
    data_path = tmp_path / "features.parquet"
    df.to_parquet(data_path, index=False)

    X = df.drop(columns=["churned"])
    y = df["churned"]
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    pipe = Pipeline([("prep", prep), ("model", estimator)]).fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)
    return data_path, model_path


def _explain(data_path, model_path, best_model):
    return explain_model(
        data_path=data_path, model_path=model_path, target_column="churned",
        problem_type="classification", best_model=best_model,
    )


def test_tree_shap_path(tmp_path: Path):
    data, model = _build_pipeline(tmp_path, RandomForestClassifier(n_estimators=40, random_state=42))
    rep = _explain(data, model, "random_forest")

    assert rep["method"] == "tree_shap"
    assert len(rep["top_features"]) > 0
    # sorted by importance descending
    imps = [f["importance"] for f in rep["top_features"]]
    assert imps == sorted(imps, reverse=True)
    # direction is set on the SHAP path
    assert rep["top_features"][0]["direction"] in {"increases", "decreases"}


def test_linear_shap_path(tmp_path: Path):
    data, model = _build_pipeline(tmp_path, LogisticRegression(max_iter=1000, random_state=42))
    rep = _explain(data, model, "logistic_regression")
    assert rep["method"] == "linear_shap"
    assert len(rep["top_features"]) > 0


def test_deterministic(tmp_path: Path):
    data, model = _build_pipeline(tmp_path, RandomForestClassifier(n_estimators=40, random_state=42))
    a = _explain(data, model, "random_forest")
    b = _explain(data, model, "random_forest")
    assert a["top_features"] == b["top_features"]


def test_permutation_fallback_when_shap_raises(tmp_path: Path, monkeypatch):
    data, model = _build_pipeline(tmp_path, RandomForestClassifier(n_estimators=40, random_state=42))

    def boom(*a, **k):
        raise RuntimeError("shap exploded")

    monkeypatch.setattr(explain_mod.shap, "TreeExplainer", boom)
    rep = _explain(data, model, "random_forest")

    assert rep["method"] == "permutation_importance"
    assert len(rep["top_features"]) > 0
    assert rep["warnings"]
    # permutation path does not report direction
    assert rep["top_features"][0]["direction"] is None


def test_unavailable_when_everything_fails(tmp_path: Path, monkeypatch):
    data, model = _build_pipeline(tmp_path, RandomForestClassifier(n_estimators=40, random_state=42))

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(explain_mod.shap, "TreeExplainer", boom)
    monkeypatch.setattr(
        "sklearn.inspection.permutation_importance",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("perm failed")),
    )
    # Must not raise despite both methods failing.
    rep = _explain(data, model, "random_forest")
    assert rep["method"] == "unavailable"
    assert rep["top_features"] == []
