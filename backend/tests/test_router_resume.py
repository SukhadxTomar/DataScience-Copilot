"""Runtime replan + checkpoint test.

When a capability fails at execution time, the router routes back to the
planner for a bounded replan that retries the failed step — without
re-invoking the planner LLM (the validated plan persists in checkpointed
state). This drives that path with a capability rigged to fail once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.planner.registry as reg
from app.agents.eda_agent import Insight, InsightReport
from app.agents.feature_agent import FeaturePlan
from app.agents.problem_agent import ProblemSpec
from app.agents.recommendations_agent import Recommendation, RecommendationsReport
from app.core.config import settings
from app.graph.nodes import training_node
from app.llm.fake import FakeLLMClient
from app.planner.planner_agent import ExecutionPlan
from app.planner.registry import CANONICAL_PLAN
from tests.conftest import make_dataframe


@pytest.fixture
def flaky_env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "artifacts").mkdir()

    dataset_id = "testds"
    ds_dir = tmp_path / "datasets" / dataset_id
    ds_dir.mkdir()
    make_dataframe().to_csv(ds_dir / "raw.csv", index=False)
    (ds_dir / "metadata.json").write_text(json.dumps({"dataset_id": dataset_id}))

    # training fails on its first call, succeeds on the retry.
    calls = {"training": 0}

    def flaky_training(state):
        calls["training"] += 1
        if calls["training"] == 1:
            raise RuntimeError("transient training failure")
        return training_node(state)

    flaky_cap = reg.Capability(
        name="training",
        requires=reg.REGISTRY["training"].requires,
        produces=reg.REGISTRY["training"].produces,
        needs_llm=False,
        runner=flaky_training,
    )
    monkeypatch.setitem(reg.REGISTRY, "training", flaky_cap)

    fake = FakeLLMClient(responses=[
        ExecutionPlan(steps=CANONICAL_PLAN, reasoning="full pipeline"),
        InsightReport(summary="s", insights=[Insight(
            title="t", detail="d", severity="info", columns=[])], suggested_target="churned"),
        ProblemSpec(problem_type="classification", target_column="churned",
                    drop_columns=["customer_id"], drop_duplicates=False, reasoning="r"),
        FeaturePlan(datetime_columns=["signup_date"], log_transform_columns=["income"],
                    drop_columns=[], reasoning="r"),
        RecommendationsReport(narrative="ok", recommendations=[Recommendation(
            title="t", detail="d", expected_impact="i", confidence="medium")]),
    ])
    monkeypatch.setattr("app.graph.nodes.get_llm_client", lambda: fake)
    monkeypatch.setattr("app.planner.planner_node.get_llm_client", lambda: fake)

    from app.graph import builder
    builder.build_graph.cache_clear()

    def run():
        graph = builder.build_graph()
        state = {"run_id": "run001", "dataset_id": dataset_id,
                 "problem_text": "predict churn", "status": "running"}
        final = graph.invoke(
            state,
            config={"configurable": {"thread_id": "run001"}, "recursion_limit": 30},
        )
        return final, fake, calls

    yield run
    builder.build_graph.cache_clear()


def test_runtime_failure_replans_and_recovers(flaky_env):
    final, fake, calls = flaky_env()

    # The run completed despite the first training attempt failing.
    assert final["status"] == "completed"
    assert calls["training"] == 2          # failed once, retried once
    assert final["replan_count"] == 1      # exactly one replan consumed

    # The planner LLM was called exactly once — the plan persisted through the
    # replan rather than being regenerated.
    planner_calls = [c for c in fake.calls if c["schema"] == "ExecutionPlan"]
    assert len(planner_calls) == 1

    # Checkpointing is active.
    assert (settings.data_dir / "checkpoints.db").exists()
