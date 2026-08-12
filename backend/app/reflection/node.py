"""Reflection node — the diagnose -> repair -> retry policy.

This is the only place the auto-fix control policy lives. ``reflect_node`` is a
normal LangGraph node: it reads the failure the capability wrapper recorded
(``plan_error`` / ``failed_capability`` / ``failed_exc_type``), diagnoses it,
selects a repair or a planner escalation, executes an artifact repair, and
returns a partial state update that either:

- re-dispatches the SAME capability (repair applied and changed something), or
- escalates to the planner (budget exhausted, no repair mapped, repair no-op, or
  the diagnosis is a planning concern).

The plan cursor is never advanced here — it still points at the failed step, so
clearing ``plan_error`` and setting ``pending_retry_capability`` makes ``route``
re-run that same capability. Escalation sets ``escalate_to_planner`` so ``route``
hands control to the planner instead.

Three independent bounded budgets guard against loops: ``reflection_attempts``
(per failing capability), ``repair_attempts`` (per run), and the planner's own
``replan_attempts``. The graph's ``recursion_limit`` remains the final backstop.
"""

from __future__ import annotations

import copy
import logging
import time

from app.core.config import settings
from app.graph.state import RunState
from app.llm.factory import get_llm_client
from app.reflection.diagnoser import ReflectionAgent
from app.reflection.models import (
    Diagnosis,
    PlannerHint,
    ReflectionRecord,
    RepairResult,
    RepairSpec,
)
from app.reflection.repair_registry import REPAIR_REGISTRY
from app.reflection.selection import select_repair

logger = logging.getLogger(__name__)


def reflect_node(state: RunState) -> dict:
    """Diagnose the current failure and either repair-and-retry or escalate."""
    failed = state.get("failed_capability")
    error = state.get("plan_error") or ""

    # Defensive: the node should only be entered on a recorded failure. If it is
    # not, there is nothing to reflect on — escalate so the run cannot wedge.
    if not failed:
        logger.warning("reflect: entered with no failed_capability — escalating")
        return _escalate(state, reason="no failed capability to reflect on")

    attempts = dict(state.get("reflection_attempts", {}))
    used = attempts.get(failed, 0)

    # Budget check first — exhausted reflection escalates to the planner.
    if used >= settings.reflection_attempts:
        logger.info(
            "reflect: '%s' exhausted its %d reflection attempts — escalating",
            failed, settings.reflection_attempts,
        )
        return _escalate(state, reason="reflection attempts exhausted")
    if state.get("repair_attempts", 0) >= settings.repair_attempts:
        logger.info(
            "reflect: run hit the repair budget (%d) — escalating",
            settings.repair_attempts,
        )
        return _escalate(state, reason="repair budget exhausted")

    t0 = time.monotonic()
    diagnosis = ReflectionAgent(get_llm_client()).diagnose(
        failed_capability=failed,
        error=error,
        exc_type=state.get("failed_exc_type", "") or "",
        state=state,
    )
    plan = select_repair(diagnosis, state)

    # Planning concern (or nothing mapped) — hand to the planner with the hint.
    if plan.escalate or plan.spec is None:
        record = _record(failed, error, diagnosis, None, None, used + 1, t0)
        logger.info(
            "reflect: '%s' diagnosed %s -> escalating to planner",
            failed, diagnosis.category.value,
        )
        return _escalate(
            state,
            reason="planner escalation",
            record=record,
            planner_hint=plan.planner_hint,
        )

    spec = plan.spec
    result = _run_repair(spec, state, diagnosis)
    record = _record(failed, error, diagnosis, spec, result, used + 1, t0)
    attempts[failed] = used + 1

    if not result.applied or not result.changed:
        # The repair could not help — don't burn a retry on an unchanged artifact.
        logger.info(
            "reflect: repair '%s' made no change (%s) — escalating",
            spec.name, result.detail,
        )
        return _escalate(
            state, reason="repair made no change", record=record, attempts=attempts
        )

    logger.info(
        "reflect: repair '%s' applied (%s) — retrying '%s'",
        spec.name, result.detail, failed,
    )
    update: dict = {
        "plan_error": None,
        "failed_capability": None,
        "failed_exc_type": None,
        "escalate_to_planner": False,
        "pending_retry_capability": failed,  # route() re-dispatches this capability
        "reflection_attempts": attempts,
        "repair_attempts": state.get("repair_attempts", 0) + 1,
        "last_repair": result.model_dump(),
        "reflection_history": state.get("reflection_history", []) + [record.model_dump()],
    }
    # A knob repair (e.g. reduce_cv_folds) rewrites a spec value rather than a
    # dataset. Persist the knob so the retried capability honours it.
    knob = _apply_knob_repair(spec, result, state)
    if knob is not None:
        update["problem_spec"] = knob
    return update


def _run_repair(
    spec: RepairSpec, state: RunState, diagnosis: Diagnosis
) -> RepairResult:
    """Invoke a repair runner, converting an unexpected raise into a failed
    RepairResult so the node's budget accounting stays in control."""
    cap = REPAIR_REGISTRY.get(spec.name)
    if cap is None:
        return RepairResult(
            repair_name=spec.name,
            applied=False,
            changed=False,
            detail=f"repair '{spec.name}' is not registered",
            error="unknown_repair",
        )
    try:
        return cap.runner(state, diagnosis)
    except Exception as exc:  # noqa: BLE001 — a repair must not crash the graph
        logger.warning("reflect: repair '%s' raised: %s", spec.name, exc)
        return RepairResult(
            repair_name=spec.name,
            applied=False,
            changed=False,
            detail=f"repair raised: {exc}",
            error=str(exc),
        )


def _apply_knob_repair(
    spec: RepairSpec, result: RepairResult, state: RunState
) -> dict | None:
    """Materialise a knob repair into an updated ``problem_spec``, or None.

    Knob repairs signal their target via ``RepairResult.artifact_key`` with no
    ``new_artifact_path`` (they change a spec value, not a dataset on disk). The
    only knob today is ``cv_folds``; the value is parsed from the repair detail
    ("reduce cv_folds 5 -> 4"). Returns a new spec dict to merge, or None when
    the repair is a normal dataset rewrite.
    """
    if result.new_artifact_path is not None or result.artifact_key != "cv_folds":
        return None
    new_value = _parse_trailing_int(result.detail)
    if new_value is None:
        return None
    spec_dict = copy.deepcopy(state.get("problem_spec") or {})
    spec_dict["cv_folds"] = new_value
    return spec_dict


def _parse_trailing_int(detail: str) -> int | None:
    """Extract the last integer in a detail string, e.g. the 4 in '... -> 4'."""
    token = detail.strip().split()[-1] if detail.strip() else ""
    try:
        return int(token)
    except ValueError:
        return None


def _record(
    failed_capability: str,
    original_error: str,
    diagnosis: Diagnosis,
    repair: RepairSpec | None,
    repair_result: RepairResult | None,
    attempt: int,
    t0: float,
) -> ReflectionRecord:
    return ReflectionRecord(
        failed_capability=failed_capability,
        original_error=original_error,
        diagnosis=diagnosis,
        repair=repair,
        repair_result=repair_result,
        retry_succeeded=None,  # filled in by a later successful re-run, if any
        attempt=attempt,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _escalate(
    state: RunState,
    *,
    reason: str,
    record: ReflectionRecord | None = None,
    planner_hint: PlannerHint | None = None,
    attempts: dict[str, int] | None = None,
) -> dict:
    """Build the state update that hands control to the planner.

    Clears the failure so the planner's replan branch (not the reflect node) owns
    the next decision, records the reason as a warning, and appends any reflection
    record to the persisted history.
    """
    update: dict = {
        "escalate_to_planner": True,
        "pending_retry_capability": None,
        "warnings": state.get("warnings", []) + [f"reflect: {reason}"],
    }
    if attempts is not None:
        update["reflection_attempts"] = attempts
    if planner_hint is not None:
        update["planner_hint"] = planner_hint.model_dump()
    if record is not None:
        update["reflection_history"] = state.get("reflection_history", []) + [
            record.model_dump()
        ]
    return update
