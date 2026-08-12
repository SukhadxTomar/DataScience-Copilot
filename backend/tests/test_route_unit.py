"""Tests for the pure router function."""

from __future__ import annotations

from app.planner.planner_node import route

PLAN = ["insights", "problem_spec", "cleaning", "summarize"]


def test_route_returns_step_at_cursor():
    assert route({"execution_plan": PLAN, "plan_cursor": 0}) == "insights"
    assert route({"execution_plan": PLAN, "plan_cursor": 2}) == "cleaning"


def test_route_ends_when_cursor_exhausts_plan():
    assert route({"execution_plan": PLAN, "plan_cursor": len(PLAN)}) == "__end__"


def test_route_reflects_on_plan_error():
    # A fresh runtime failure goes to the reflection layer first (diagnose +
    # auto-fix), NOT straight to the planner.
    state = {"execution_plan": PLAN, "plan_cursor": 1, "plan_error": "boom"}
    assert route(state) == "reflect"


def test_route_escalates_to_planner_when_reflection_gives_up():
    # Once the reflect node sets escalate_to_planner, control goes to the planner.
    state = {
        "execution_plan": PLAN,
        "plan_cursor": 1,
        "escalate_to_planner": True,
    }
    assert route(state) == "planner"


def test_route_dispatches_pending_retry_capability():
    # After a successful repair the reflect node parks the capability to re-run.
    state = {
        "execution_plan": PLAN,
        "plan_cursor": 1,
        "pending_retry_capability": "cleaning",
    }
    assert route(state) == "cleaning"


def test_route_reflect_precedes_escalation_and_retry():
    # Precedence: a live plan_error is reflected on before any stale escalate /
    # retry flag is honoured, so a new failure is never misrouted to the planner.
    state = {
        "execution_plan": PLAN,
        "plan_cursor": 1,
        "plan_error": "boom",
        "escalate_to_planner": True,
        "pending_retry_capability": "cleaning",
    }
    assert route(state) == "reflect"


def test_needs_input_ends_immediately():
    state = {"execution_plan": PLAN, "plan_cursor": 1, "status": "needs_input"}
    assert route(state) == "__end__"


def test_missing_plan_ends():
    assert route({}) == "__end__"
