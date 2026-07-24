"""Tests for the planner agent (LLM stubbed with FakeLLMClient)."""

from __future__ import annotations

from app.llm.fake import FakeLLMClient
from app.planner.planner_agent import ExecutionPlan, PlannerAgent
from app.planner.registry import CANONICAL_PLAN

PROFILE = {"n_rows": 100, "n_cols": 5, "columns": [{"name": "churned"}]}


def test_returns_the_planned_steps():
    fake = FakeLLMClient(responses=[
        ExecutionPlan(steps=CANONICAL_PLAN, reasoning="full pipeline"),
    ])
    plan = PlannerAgent(fake).run(PROFILE, "predict churn")
    assert plan.steps == CANONICAL_PLAN
    assert fake.calls[0]["schema"] == "ExecutionPlan"


def test_hallucinated_names_are_dropped():
    fake = FakeLLMClient(responses=[
        ExecutionPlan(
            steps=["insights", "made_up", "summarize", "insights"],
            reasoning="mix of real, invented, and duplicate",
        ),
    ])
    plan = PlannerAgent(fake).run(PROFILE, "goal")
    # made_up dropped, duplicate insights collapsed, order preserved.
    assert plan.steps == ["insights", "summarize"]


def test_prompt_is_registry_bound():
    fake = FakeLLMClient(responses=[
        ExecutionPlan(steps=CANONICAL_PLAN, reasoning="r"),
    ])
    PlannerAgent(fake).run(PROFILE, "goal")
    user_prompt = fake.calls[0]["user"]
    # The available capabilities and their requires/produces are in the prompt.
    assert "insights" in user_prompt
    assert "requires" in user_prompt and "produces" in user_prompt


def test_feedback_is_appended_on_retry():
    fake = FakeLLMClient(responses=[
        ExecutionPlan(steps=CANONICAL_PLAN, reasoning="r"),
    ])
    PlannerAgent(fake).run(PROFILE, "goal", feedback="step 3 'features' needs feature_plan")
    assert "features" in fake.calls[0]["user"]
    assert "rejected by the validator" in fake.calls[0]["user"]
