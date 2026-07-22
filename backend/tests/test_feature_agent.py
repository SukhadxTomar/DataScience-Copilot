"""Tests for the feature engineering agent.

The agent's own logic is the validation/filtering layer around the LLM
output; the LLM is replaced with a FakeLLMClient returning canned plans.
"""

from __future__ import annotations

from pathlib import Path

from app.agents.feature_agent import FeatureEngineeringAgent, FeaturePlan
from app.llm.fake import FakeLLMClient
from app.tools.profiler import profile_dataset


def _profile(cleaned_parquet: Path) -> dict:
    return profile_dataset(cleaned_parquet)


def test_returns_plan_and_records_call(cleaned_parquet: Path):
    fake = FakeLLMClient(responses=[
        FeaturePlan(
            datetime_columns=["signup_date"],
            log_transform_columns=["income"],
            drop_columns=[],
            reasoning="signup_date carries seasonality; income is right-skewed.",
        )
    ])
    agent = FeatureEngineeringAgent(llm=fake)
    spec = {"problem_type": "classification", "target_column": "churned"}

    plan = agent.run(spec, _profile(cleaned_parquet))

    assert plan.datetime_columns == ["signup_date"]
    assert plan.log_transform_columns == ["income"]
    assert fake.calls[0]["schema"] == "FeaturePlan"


def test_filters_hallucinated_columns(cleaned_parquet: Path):
    fake = FakeLLMClient(responses=[
        FeaturePlan(
            datetime_columns=["signup_date", "does_not_exist"],
            log_transform_columns=["income", "ghost_col"],
            drop_columns=["also_missing"],
            reasoning="mix of real and invented columns",
        )
    ])
    agent = FeatureEngineeringAgent(llm=fake)
    spec = {"problem_type": "classification", "target_column": "churned"}

    plan = agent.run(spec, _profile(cleaned_parquet))

    assert plan.datetime_columns == ["signup_date"]
    assert plan.log_transform_columns == ["income"]
    assert plan.drop_columns == []


def test_never_keeps_target_in_any_list(cleaned_parquet: Path):
    fake = FakeLLMClient(responses=[
        FeaturePlan(
            datetime_columns=["churned"],
            log_transform_columns=["churned"],
            drop_columns=["churned"],
            reasoning="agent mistakenly targeted the target",
        )
    ])
    agent = FeatureEngineeringAgent(llm=fake)
    spec = {"problem_type": "classification", "target_column": "churned"}

    plan = agent.run(spec, _profile(cleaned_parquet))

    assert "churned" not in plan.datetime_columns
    assert "churned" not in plan.log_transform_columns
    assert "churned" not in plan.drop_columns


def test_deduplicates_columns(cleaned_parquet: Path):
    fake = FakeLLMClient(responses=[
        FeaturePlan(
            datetime_columns=["signup_date", "signup_date"],
            log_transform_columns=[],
            drop_columns=[],
            reasoning="duplicate entry",
        )
    ])
    agent = FeatureEngineeringAgent(llm=fake)
    spec = {"problem_type": "classification", "target_column": "churned"}

    plan = agent.run(spec, _profile(cleaned_parquet))
    assert plan.datetime_columns == ["signup_date"]
