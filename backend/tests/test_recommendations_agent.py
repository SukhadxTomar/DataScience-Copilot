"""Tests for the recommendations agent (LLM stubbed)."""

from __future__ import annotations

from app.agents.recommendations_agent import (
    Recommendation,
    RecommendationsAgent,
    RecommendationsReport,
)
from app.llm.fake import FakeLLMClient

SPEC = {"problem_type": "classification", "target_column": "churned"}
EVAL = {"metrics": {"accuracy": 0.83, "roc_auc": 0.88}}
EXPLAIN = {"method": "tree_shap", "top_features": [
    {"feature": "income", "importance": 0.5, "direction": "increases"},
    {"feature": "age", "importance": 0.3, "direction": "decreases"},
]}


def _rec(title="Act on income"):
    return Recommendation(title=title, detail="d", expected_impact="i", confidence="high")


def test_returns_report_and_records_call():
    fake = FakeLLMClient(responses=[
        RecommendationsReport(narrative="Solid model.", recommendations=[_rec()]),
    ])
    rep = RecommendationsAgent(fake).run(SPEC, EVAL, EXPLAIN)
    assert rep.narrative == "Solid model."
    assert rep.recommendations[0].title == "Act on income"
    assert fake.calls[0]["schema"] == "RecommendationsReport"


def test_caps_number_of_recommendations():
    many = [_rec(f"rec {i}") for i in range(10)]
    fake = FakeLLMClient(responses=[RecommendationsReport(narrative="n", recommendations=many)])
    rep = RecommendationsAgent(fake).run(SPEC, EVAL, EXPLAIN)
    assert len(rep.recommendations) == 6  # MAX_RECOMMENDATIONS


def test_prompt_is_grounded_in_metrics_and_features():
    fake = FakeLLMClient(responses=[RecommendationsReport(narrative="n", recommendations=[_rec()])])
    RecommendationsAgent(fake).run(SPEC, EVAL, EXPLAIN)
    user = fake.calls[0]["user"]
    assert "accuracy" in user and "income" in user
