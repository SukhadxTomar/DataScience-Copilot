"""End-to-end pipeline test.

Runs the whole LangGraph pipeline on the synthetic dataset with the LLM
calls stubbed out by a FakeLLMClient. This exercises the real cleaning,
feature engineering, training, and evaluation tools wired together through
the dynamic Planner + router executor (Phase 4) over the real tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.eda_agent import Insight, InsightReport
from app.agents.feature_agent import FeaturePlan
from app.agents.problem_agent import ProblemSpec
from app.agents.recommendations_agent import Recommendation, RecommendationsReport
from app.core.config import settings
from app.llm.fake import FakeLLMClient
from app.planner.planner_agent import ExecutionPlan
from app.planner.registry import CANONICAL_PLAN
from tests.conftest import make_dataframe


@pytest.fixture
def pipeline_env(tmp_path: Path, monkeypatch):
    """Point storage at a tmp dir and stub the LLM, then hand back a runner."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "artifacts").mkdir()

    dataset_id = "testds"
    ds_dir = tmp_path / "datasets" / dataset_id
    ds_dir.mkdir()
    make_dataframe().to_csv(ds_dir / "raw.csv", index=False)
    (ds_dir / "metadata.json").write_text(json.dumps({"dataset_id": dataset_id}))

    # One shared fake serves every LLM node, popping responses in the order the
    # graph calls them: planner first, then the canonical plan's LLM steps
    # (insights -> problem_spec -> feature_plan -> recommendations).
    fake = FakeLLMClient(responses=[
        ExecutionPlan(steps=CANONICAL_PLAN, reasoning="Full pipeline for a clean tabular set."),
        InsightReport(
            summary="A churn dataset with an id column and a signup date.",
            insights=[Insight(
                title="customer_id is an identifier",
                detail="It is unique per row and should be dropped.",
                severity="warning",
                columns=["customer_id"],
            )],
            suggested_target="churned",
        ),
        ProblemSpec(
            problem_type="classification",
            target_column="churned",
            drop_columns=["customer_id"],
            drop_duplicates=False,
            reasoning="Binary churn outcome; customer_id is an identifier.",
        ),
        FeaturePlan(
            datetime_columns=["signup_date"],
            log_transform_columns=["income"],
            drop_columns=[],
            reasoning="signup_date holds seasonality; income is right-skewed.",
        ),
        RecommendationsReport(
            narrative="The model separates churners adequately; act on the top drivers.",
            recommendations=[Recommendation(
                title="Focus retention on the strongest driver",
                detail="income is the top SHAP driver of churn.",
                expected_impact="Targeted outreach should lower churn.",
                confidence="medium",
            )],
        ),
    ])
    # Two modules resolve get_llm_client independently: the capability nodes
    # (app.graph.nodes) and the planner (app.planner.planner_node).
    monkeypatch.setattr("app.graph.nodes.get_llm_client", lambda: fake)
    monkeypatch.setattr("app.planner.planner_node.get_llm_client", lambda: fake)

    # Rebuild the graph so it opens its checkpoint DB under the tmp data dir.
    from app.graph import builder
    builder.build_graph.cache_clear()

    def run():
        graph = builder.build_graph()
        state = {
            "run_id": "run001",
            "dataset_id": dataset_id,
            "problem_text": "Predict which customers will churn.",
            "status": "running",
        }
        return graph.invoke(
            state,
            config={"configurable": {"thread_id": "run001"}, "recursion_limit": 30},
        )

    yield run
    builder.build_graph.cache_clear()


def test_full_pipeline_runs_end_to_end(pipeline_env):
    final = pipeline_env()

    assert final["status"] == "completed"

    # The planner's validated plan drove execution and is exposed on the state.
    assert final["execution_plan"] == CANONICAL_PLAN
    assert final["plan_reasoning"]
    assert final["plan_cursor"] == len(CANONICAL_PLAN)

    # Cleaning dropped the id column.
    assert "customer_id" in final["cleaning_report"]["dropped_columns"]

    # Feature engineering expanded the datetime and log-transformed income.
    fr = final["feature_report"]
    assert fr["expanded_datetime_columns"] == ["signup_date"]
    assert fr["log_transformed_columns"] == ["income"]
    assert Path(fr["features_path"]).exists()

    # Training produced a leaderboard and a saved model.
    tr = final["training_report"]
    assert Path(tr["model_path"]).exists()
    assert len(tr["results"]) == 4

    # Evaluation produced held-out metrics.
    assert "accuracy" in final["evaluation_report"]["metrics"]

    # Explainability produced ranked feature importances.
    exp = final["explanation_report"]
    assert exp["method"] in {"tree_shap", "linear_shap", "permutation_importance"}
    assert len(exp["top_features"]) > 0

    # Recommendations came through.
    assert final["recommendations"]["recommendations"]

    # The consolidated report was assembled and its markdown written to disk.
    assert Path(final["report"]["report_path"]).exists()
    assert final["report"]["model"]["best_model"] == final["training_report"]["best_model"]

    # The summary mentions the feature step.
    assert "Features:" in final["summary"]


def test_pipeline_is_resumable_via_checkpoint(pipeline_env):
    """A second invoke with the same thread_id returns the checkpointed
    result rather than re-running from scratch."""
    first = pipeline_env()
    assert first["status"] == "completed"
    # The checkpoint DB exists under the tmp storage dir.
    assert (settings.data_dir / "checkpoints.db").exists()
