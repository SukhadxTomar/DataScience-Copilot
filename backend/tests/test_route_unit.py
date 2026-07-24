"""Tests for the pure router function."""

from __future__ import annotations

from app.planner.planner_node import route

PLAN = ["insights", "problem_spec", "cleaning", "summarize"]


def test_route_returns_step_at_cursor():
    assert route({"execution_plan": PLAN, "plan_cursor": 0}) == "insights"
    assert route({"execution_plan": PLAN, "plan_cursor": 2}) == "cleaning"


def test_route_ends_when_cursor_exhausts_plan():
    assert route({"execution_plan": PLAN, "plan_cursor": len(PLAN)}) == "__end__"


def test_route_replans_on_plan_error():
    state = {"execution_plan": PLAN, "plan_cursor": 1, "plan_error": "boom"}
    assert route(state) == "planner"


def test_needs_input_ends_immediately():
    state = {"execution_plan": PLAN, "plan_cursor": 1, "status": "needs_input"}
    assert route(state) == "__end__"


def test_missing_plan_ends():
    assert route({}) == "__end__"
