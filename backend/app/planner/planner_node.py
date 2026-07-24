"""Planner node, retry loop, and router.

This module turns the planner agent + validator into the executable control
layer of the graph:

- ``generate_plan`` runs the planner up to ``plan_attempts`` times, feeding
  each validation failure back to the model, and gives up into a
  ``needs_input`` outcome rather than running an unsound plan.
- ``planner_node`` is the graph node: it produces the validated plan on first
  entry, and on a runtime failure it is re-entered as a bounded *replan*
  (retrying the failed step up to ``replan_attempts`` times before stopping
  for human input).
- ``route`` is the pure conditional-edge function the executor uses to walk
  the plan one capability at a time, checkpointing after each.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.graph.state import RunState
from app.llm.base import LLMError
from app.llm.factory import get_llm_client
from app.planner.planner_agent import PlannerAgent
from app.planner.registry import BOOTSTRAP_ARTIFACTS, REGISTRY
from app.planner.validator import validate_plan

logger = logging.getLogger(__name__)

# Every state key any capability can produce — used to detect which work is
# already done when (re)planning.
_ALL_PRODUCES: set[str] = set().union(*(cap.produces for cap in REGISTRY.values()))


def _produced_artifacts(state: RunState) -> set[str]:
    """Bootstrap artifacts plus any capability outputs already in the state."""
    produced = set(BOOTSTRAP_ARTIFACTS)
    for key in _ALL_PRODUCES:
        if state.get(key) is not None:
            produced.add(key)
    return produced


def generate_plan(
    profile: dict,
    problem_text: str,
    initial_artifacts: set[str],
) -> tuple[list[str] | None, str, list[str], list[str]]:
    """Plan, validate, and retry up to ``plan_attempts`` times.

    Returns ``(steps, reasoning, warnings, errors)``. On success ``steps`` is
    the validated plan and ``errors`` is empty; if every attempt fails,
    ``steps`` is ``None`` and ``errors`` holds the final validator (or LLM)
    errors for a human to act on.
    """
    warnings: list[str] = []
    feedback: str | None = None
    errors: list[str] = []

    for attempt in range(settings.plan_attempts):
        try:
            plan = PlannerAgent(get_llm_client()).run(profile, problem_text, feedback)
        except LLMError as exc:
            errors = [f"LLM error during planning: {exc}"]
            feedback = "\n".join(errors)
            warnings.append(f"plan attempt {attempt + 1} failed: {exc}")
            continue

        ok, errors = validate_plan(plan.steps, initial_artifacts, REGISTRY)
        if ok:
            return plan.steps, plan.reasoning, warnings, []
        feedback = "\n".join(errors)
        warnings.append(f"plan attempt {attempt + 1} invalid: {feedback}")

    return None, "", warnings, errors


def planner_node(state: RunState) -> dict:
    """Produce the execution plan, or handle a bounded runtime replan."""
    # A replan is triggered when a capability runner failed at execution time
    # (the wrapper set ``plan_error`` without advancing the cursor).
    if state.get("plan_error"):
        failed = state.get("failed_capability")
        replan_count = state.get("replan_count", 0) + 1
        if replan_count > settings.replan_attempts:
            logger.warning(
                "run=%s giving up after %d replans; '%s' keeps failing",
                state["run_id"], settings.replan_attempts, failed,
            )
            return {
                "status": "needs_input",
                "plan_error": None,
                "errors": [
                    f"capability '{failed}' failed after "
                    f"{settings.replan_attempts} replans: {state['plan_error']}"
                ],
            }
        logger.info(
            "run=%s replan %d/%d, retrying after '%s' failed",
            state["run_id"], replan_count, settings.replan_attempts, failed,
        )
        # Retry the current plan from the failed step: keep execution_plan and
        # plan_cursor as-is, just clear the error so the router resumes there.
        return {
            "replan_count": replan_count,
            "plan_error": None,
            "failed_capability": None,
            "warnings": state.get("warnings", [])
            + [f"replan {replan_count}: retrying after '{failed}' failed"],
        }

    # First-time planning.
    steps, reasoning, warnings, errors = generate_plan(
        profile=state["profile"],
        problem_text=state.get("problem_text", ""),
        initial_artifacts=_produced_artifacts(state),
    )
    if steps is None:
        logger.warning(
            "run=%s no valid plan after %d attempts -> needs_input",
            state["run_id"], settings.plan_attempts,
        )
        return {
            "status": "needs_input",
            "execution_plan": [],
            "plan_cursor": 0,
            "warnings": warnings,
            "errors": errors,
        }

    logger.info("run=%s plan: %s", state["run_id"], steps)
    return {
        "execution_plan": steps,
        "plan_reasoning": reasoning,
        "plan_cursor": 0,
        "replan_count": 0,
        "warnings": warnings,
    }


def route(state: RunState) -> str:
    """Pure router: pick the next node from the plan and cursor.

    Returns a capability name (== node name), ``"planner"`` to replan after a
    runtime failure, or ``"__end__"`` when the plan is done or the run has
    stopped for human input.
    """
    if state.get("status") == "needs_input":
        return "__end__"
    if state.get("plan_error"):
        return "planner"
    plan = state.get("execution_plan") or []
    cursor = state.get("plan_cursor", 0)
    if cursor >= len(plan):
        return "__end__"
    return plan[cursor]
