"""Tests for the capability registry."""

from __future__ import annotations

from app.planner.registry import (
    CANONICAL_PLAN,
    REGISTRY,
    all_names,
    describe_for_prompt,
    get,
)


def test_all_runners_are_callable():
    for cap in REGISTRY.values():
        assert callable(cap.runner)


def test_produces_are_pairwise_disjoint():
    """No two capabilities produce the same artifact — a precondition for the
    validator's duplicate-producer check to be meaningful."""
    seen: dict[str, str] = {}
    for cap in REGISTRY.values():
        for key in cap.produces:
            assert key not in seen, f"{key} produced by both {seen.get(key)} and {cap.name}"
            seen[key] = cap.name


def test_canonical_plan_covers_every_capability_once():
    assert sorted(CANONICAL_PLAN) == sorted(all_names())
    assert len(CANONICAL_PLAN) == len(set(CANONICAL_PLAN))


def test_describe_for_prompt_lists_all_capabilities():
    text = describe_for_prompt()
    for name in all_names():
        assert name in text


def test_insights_not_required_by_summarize():
    """insights must stay optional for the planner to have a real choice."""
    assert "insights" not in get("summarize").requires


def test_profile_is_not_a_capability():
    assert "profile" not in REGISTRY
