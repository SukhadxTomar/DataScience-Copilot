"""Tests for the bounded plan-generation retry loop."""

from __future__ import annotations

from app.planner import planner_node as pn
from app.llm.fake import FakeLLMClient
from app.planner.planner_agent import ExecutionPlan
from app.planner.registry import BOOTSTRAP_ARTIFACTS, CANONICAL_PLAN

PROFILE = {"n_rows": 100, "columns": [{"name": "churned"}]}
BOOTSTRAP = set(BOOTSTRAP_ARTIFACTS)

# A plan that sanitizes cleanly (registered names) but fails the validator.
INVALID = ["features", "summarize"]


def _use_fake(monkeypatch, responses):
    fake = FakeLLMClient(responses=responses)
    monkeypatch.setattr(pn, "get_llm_client", lambda: fake)
    return fake


def test_invalid_then_valid_retries_and_succeeds(monkeypatch):
    fake = _use_fake(monkeypatch, [
        ExecutionPlan(steps=INVALID, reasoning="bad"),
        ExecutionPlan(steps=CANONICAL_PLAN, reasoning="good"),
    ])
    steps, reasoning, warnings, errors = pn.generate_plan(PROFILE, "goal", BOOTSTRAP)

    assert steps == CANONICAL_PLAN
    assert errors == []
    assert len(fake.calls) == 2           # one retry consumed
    assert any("invalid" in w for w in warnings)


def test_all_invalid_gives_up_with_errors(monkeypatch):
    fake = _use_fake(monkeypatch, [
        ExecutionPlan(steps=INVALID, reasoning="bad") for _ in range(3)
    ])
    steps, reasoning, warnings, errors = pn.generate_plan(PROFILE, "goal", BOOTSTRAP)

    assert steps is None                  # exhausted -> caller turns into needs_input
    assert errors                         # validator errors surfaced
    assert len(fake.calls) == 3           # settings.plan_attempts


def test_llm_error_is_bounded_too(monkeypatch):
    # FakeLLMClient with no responses raises LLMError on every call.
    fake = _use_fake(monkeypatch, [])
    steps, reasoning, warnings, errors = pn.generate_plan(PROFILE, "goal", BOOTSTRAP)

    assert steps is None
    assert any("LLM error" in e for e in errors)
    assert len(fake.calls) == 3


def test_planner_node_needs_input_when_no_valid_plan(monkeypatch):
    _use_fake(monkeypatch, [ExecutionPlan(steps=INVALID, reasoning="bad") for _ in range(3)])
    out = pn.planner_node({"run_id": "r1", "profile": PROFILE, "problem_text": "goal"})

    assert out["status"] == "needs_input"
    assert out["execution_plan"] == []
    assert out["errors"]


def test_planner_node_replan_retries_then_stops(monkeypatch):
    # No LLM needed on the replan path — it retries the existing plan.
    base = {
        "run_id": "r1",
        "execution_plan": CANONICAL_PLAN,
        "plan_cursor": 4,
        "plan_error": "training blew up",
        "failed_capability": "training",
    }
    # First replan (count 0 -> 1): clears error, keeps going.
    out1 = pn.planner_node({**base, "replan_count": 0})
    assert out1["replan_count"] == 1
    assert out1["plan_error"] is None
    assert "status" not in out1

    # Budget is replan_attempts=2; the third failure (count 2 -> 3) stops.
    out_stop = pn.planner_node({**base, "replan_count": 2})
    assert out_stop["status"] == "needs_input"
    assert any("training" in e for e in out_stop["errors"])
