"""End-to-end pipeline test.

Runs the whole LangGraph pipeline on the synthetic dataset with the LLM
calls stubbed out by a FakeLLMClient. This exercises the real cleaning,
feature engineering, training, and evaluation tools wired together through
the graph — the integration Phase 3 is meant to deliver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.eda_agent import Insight, InsightReport
from app.agents.feature_agent import FeaturePlan
from app.agents.problem_agent import ProblemSpec
from app.core.config import settings
from app.llm.fake import FakeLLMClient
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

    # One shared fake serves all three LLM nodes, popping responses in the
    # order the graph calls them: insights -> problem_spec -> feature_plan.
    fake = FakeLLMClient(responses=[
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
    ])
    monkeypatch.setattr("app.graph.nodes.get_llm_client", lambda: fake)

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
        return graph.invoke(state, config={"configurable": {"thread_id": "run001"}})

    yield run
    builder.build_graph.cache_clear()


def test_full_pipeline_runs_end_to_end(pipeline_env):
    final = pipeline_env()

    assert final["status"] == "completed"

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

    # The summary mentions the feature step.
    assert "Features:" in final["summary"]


def test_pipeline_is_resumable_via_checkpoint(pipeline_env):
    """A second invoke with the same thread_id returns the checkpointed
    result rather than re-running from scratch."""
    first = pipeline_env()
    assert first["status"] == "completed"
    # The checkpoint DB exists under the tmp storage dir.
    assert (settings.data_dir / "checkpoints.db").exists()
