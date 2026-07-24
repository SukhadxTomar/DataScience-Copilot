"""Tests for the DAG validator — pure, no LLM, no I/O."""

from __future__ import annotations

from app.planner.registry import BOOTSTRAP_ARTIFACTS, CANONICAL_PLAN, REGISTRY
from app.planner.validator import validate_plan

BOOTSTRAP = set(BOOTSTRAP_ARTIFACTS)


def test_canonical_plan_is_valid():
    ok, errors = validate_plan(CANONICAL_PLAN, BOOTSTRAP, REGISTRY)
    assert ok
    assert errors == []


def test_unknown_capability_rejected():
    ok, errors = validate_plan(
        ["insights", "not_a_real_step", "summarize"], BOOTSTRAP, REGISTRY
    )
    assert not ok
    assert any("not_a_real_step" in e for e in errors)


def test_out_of_order_missing_requires():
    # features before cleaning/feature_plan -> its requires are unmet.
    plan = ["problem_spec", "features", "cleaning", "feature_plan",
            "training", "evaluation", "summarize"]
    ok, errors = validate_plan(plan, BOOTSTRAP, REGISTRY)
    assert not ok
    assert any("features" in e and "no earlier step produces" in e for e in errors)


def test_missing_terminal_artifacts():
    ok, errors = validate_plan(
        ["insights", "problem_spec", "cleaning"], BOOTSTRAP, REGISTRY
    )
    assert not ok
    joined = " ".join(errors)
    assert "summary" in joined and "report" in joined


def test_duplicate_producer_rejected():
    plan = ["problem_spec", "cleaning", "cleaning", "feature_plan", "features",
            "training", "evaluation", "explain", "recommendations", "summarize", "report"]
    ok, errors = validate_plan(plan, BOOTSTRAP, REGISTRY)
    assert not ok
    assert any("re-produces" in e and "cleaning_report" in e for e in errors)


def test_empty_plan_rejected():
    ok, errors = validate_plan([], BOOTSTRAP, REGISTRY)
    assert not ok
    assert any("empty" in e for e in errors)


def test_insights_is_optional_and_position_free():
    """The one real degree of freedom: a plan is valid with or without
    insights, and wherever insights sits."""
    tail = ["problem_spec", "cleaning", "feature_plan", "features", "training",
            "evaluation", "explain", "recommendations", "summarize", "report"]

    # Without insights at all — still valid.
    ok, errors = validate_plan(tail, BOOTSTRAP, REGISTRY)
    assert ok, errors

    # insights floated to just before summarize is still valid.
    floated = ["problem_spec", "cleaning", "feature_plan", "features", "training",
               "evaluation", "explain", "recommendations", "insights", "summarize", "report"]
    ok, errors = validate_plan(floated, BOOTSTRAP, REGISTRY)
    assert ok, errors
