"""Tests for the final report assembler."""

from __future__ import annotations

from pathlib import Path

from app.tools.report import assemble_report

BASE_STATE = {
    "run_id": "run001",
    "dataset_id": "ds1",
    "profile": {"n_rows": 120, "n_cols": 6},
    "problem_spec": {"problem_type": "classification", "target_column": "churned"},
    "cleaning_report": {"rows_removed": 3, "dropped_columns": ["customer_id"]},
    "feature_report": {
        "n_features_before": 4, "n_features_after": 6,
        "expanded_datetime_columns": ["signup_date"], "log_transformed_columns": ["income"],
    },
    "training_report": {
        "best_model": "xgboost", "metric": "roc_auc",
        "results": [{"model": "xgboost", "cv_mean": 0.88, "cv_std": 0.02}],
    },
    "evaluation_report": {"metrics": {"accuracy": 0.83, "roc_auc": 0.88}},
    "explanation_report": {
        "method": "tree_shap",
        "top_features": [{"feature": "income", "importance": 0.5, "direction": "increases"}],
    },
    "recommendations": {
        "narrative": "Solid model.",
        "recommendations": [{
            "title": "Target high-income churners", "detail": "d",
            "expected_impact": "lower churn", "confidence": "high",
        }],
    },
    "summary": "Dataset: 120 rows x 6 columns.\nFeatures: 4 -> 6 columns.",
}


def test_assemble_writes_markdown_and_returns_dict(tmp_path: Path):
    report = assemble_report({**BASE_STATE, "insights": None}, tmp_path)

    # structured dict has all sections
    for key in ("problem", "dataset_overview", "cleaning", "features", "model",
                "evaluation", "explainability", "recommendations", "report_path"):
        assert key in report

    md_path = Path(report["report_path"])
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "xgboost" in text                       # best model
    assert "accuracy" in text                       # a metric
    assert "income" in text                         # a top feature
    assert "Target high-income churners" in text    # a recommendation


def test_insights_section_included_when_present(tmp_path: Path):
    insights = {"summary": "s", "insights": [
        {"title": "customer_id is an id", "detail": "drop it", "severity": "warning"}]}
    report = assemble_report({**BASE_STATE, "insights": insights}, tmp_path)
    text = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "Data quality findings" in text
    assert "customer_id is an id" in text


def test_graceful_without_insights_or_features(tmp_path: Path):
    # insights absent and empty top_features must not crash rendering.
    state = {**BASE_STATE, "insights": None}
    state["explanation_report"] = {"method": "unavailable", "top_features": []}
    report = assemble_report(state, tmp_path)
    text = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "Data quality findings" not in text
    assert "unavailable" in text
