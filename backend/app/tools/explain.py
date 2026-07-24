"""Model explainability tool.

Turns the trained pipeline into global feature importances — which columns
drive the model's predictions — as plain JSON the frontend can chart. SHAP is
the primary method (TreeExplainer for tree models, LinearExplainer for linear
ones); if SHAP is unavailable or errors on a given model, we fall back to
sklearn's model-agnostic permutation importance. As a last resort the tool
reports ``method="unavailable"`` with an empty list.

Design rule: **this tool never raises.** Explainability is a nice-to-have on
top of a finished model, so a SHAP/numba hiccup must not fail an otherwise
complete run (which would trip the router's replan path). Every failure mode
degrades to a lesser-but-valid report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Guarded import: keep the module importable even if shap (which pulls numba)
# will not install on this platform. The tool then uses permutation importance.
try:
    import shap  # type: ignore

    _HAS_SHAP = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_SHAP = False

RANDOM_STATE = 42
_TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}
_LINEAR_MODELS = {"logistic_regression", "ridge"}


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    direction: Literal["increases", "decreases"] | None = Field(
        default=None,
        description="Whether higher values push the prediction up or down "
        "(SHAP only; None for permutation importance)",
    )


class ExplanationReport(BaseModel):
    method: Literal["tree_shap", "linear_shap", "permutation_importance", "unavailable"]
    model: str
    top_features: list[FeatureImportance] = Field(default_factory=list)
    n_samples_explained: int = 0
    notes: str = ""
    warnings: list[str] = Field(default_factory=list)


def explain_model(
    data_path: Path,
    model_path: Path,
    target_column: str,
    problem_type: str,
    best_model: str,
    top_n: int = 15,
    sample_size: int = 200,
) -> dict[str, Any]:
    """Compute global feature importances for the trained pipeline."""
    df = pd.read_parquet(data_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Deterministic, bounded sample keeps SHAP fast on any dataset size.
    if len(X) > sample_size:
        idx = X.sample(sample_size, random_state=RANDOM_STATE).index
        Xs, ys = X.loc[idx], y.loc[idx]
    else:
        Xs, ys = X, y

    pipeline = joblib.load(model_path)
    warnings: list[str] = []

    if _HAS_SHAP:
        try:
            return _shap_report(pipeline, Xs, best_model, top_n).model_dump()
        except Exception as exc:  # noqa: BLE001 — degrade, never raise
            warnings.append(f"SHAP failed ({exc!r}); using permutation importance")
    else:
        warnings.append("shap not installed; using permutation importance")

    try:
        return _permutation_report(pipeline, Xs, ys, best_model, top_n, warnings).model_dump()
    except Exception as exc:  # noqa: BLE001 — last resort
        warnings.append(f"permutation importance failed ({exc!r})")
        return ExplanationReport(
            method="unavailable", model=best_model, warnings=warnings
        ).model_dump()


def _feature_names(preprocessor: Any) -> list[str]:
    """Human-readable names after preprocessing, without the num__/cat__ prefix."""
    names = preprocessor.get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else str(n) for n in names]


def _shap_report(pipeline: Any, Xs: pd.DataFrame, best_model: str, top_n: int) -> ExplanationReport:
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    Xt = prep.transform(Xs)
    names = _feature_names(prep)

    if best_model in _TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        method = "tree_shap"
    elif best_model in _LINEAR_MODELS:
        explainer = shap.LinearExplainer(model, Xt)
        method = "linear_shap"
    else:
        raise ValueError(f"no SHAP explainer mapped for model '{best_model}'")

    values = _normalize_shap(explainer.shap_values(Xt))  # -> (n_samples, n_features)
    importance = np.abs(values).mean(axis=0)
    signed = values.mean(axis=0)

    order = sorted(range(len(names)), key=lambda i: (-importance[i], names[i]))
    top = [
        FeatureImportance(
            feature=names[i],
            importance=round(float(importance[i]), 6),
            direction="increases" if signed[i] >= 0 else "decreases",
        )
        for i in order[:top_n]
    ]
    return ExplanationReport(
        method=method, model=best_model, top_features=top,
        n_samples_explained=len(Xs),
        notes="Global importance = mean(|SHAP value|) over the sample.",
    )


def _normalize_shap(values: Any) -> np.ndarray:
    """Reduce SHAP output to a 2-D (samples, features) array for one output.

    Handles the shape differences across model types / shap versions: a list of
    per-class arrays, or a 3-D (samples, features, classes) array — both reduced
    to the positive/last class; a plain 2-D array is returned as-is.
    """
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _permutation_report(
    pipeline: Any, Xs: pd.DataFrame, ys: pd.Series, best_model: str,
    top_n: int, warnings: list[str],
) -> ExplanationReport:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, Xs, ys, n_repeats=5, random_state=RANDOM_STATE
    )
    names = list(Xs.columns)
    importance = result.importances_mean

    order = sorted(range(len(names)), key=lambda i: (-importance[i], names[i]))
    top = [
        FeatureImportance(feature=names[i], importance=round(float(importance[i]), 6))
        for i in order[:top_n]
    ]
    return ExplanationReport(
        method="permutation_importance", model=best_model, top_features=top,
        n_samples_explained=len(Xs),
        notes="Importance = mean drop in model score when the column is shuffled.",
        warnings=warnings,
    )
