"""Shared pipeline state.

The state carries file paths, summaries, and results between graph
nodes — never dataframes or model objects. Large artifacts live on disk;
the state stays small and serializable, which is what makes checkpointing
after every node cheap.
"""

from __future__ import annotations

from typing import Any, TypedDict


class RunState(TypedDict, total=False):
    # Inputs
    run_id: str
    dataset_id: str
    problem_text: str

    # Node outputs
    profile: dict[str, Any]
    insights: dict[str, Any]
    problem_spec: dict[str, Any]     # problem type, target, columns to drop
    cleaning_report: dict[str, Any]  # what cleaning changed, cleaned data path
    feature_plan: dict[str, Any]     # which columns to expand/log-transform/drop
    feature_report: dict[str, Any]   # what feature engineering changed, features path
    training_report: dict[str, Any]  # per-model CV scores, best model path
    evaluation_report: dict[str, Any]  # held-out test metrics
    explanation_report: dict[str, Any]  # global feature importances (SHAP/fallback)
    recommendations: dict[str, Any]  # business recommendations
    report: dict[str, Any]           # consolidated final report + report path
    summary: str

    # Planning / execution control (Phase 4)
    execution_plan: list[str]        # validated, ordered capability names
    plan_reasoning: str              # planner's justification
    plan_cursor: int                 # index of the next capability to run
    replan_count: int                # runtime replans consumed so far
    warnings: list[str]              # non-fatal notices (dropped names, fallbacks)
    errors: list[str]                # validator errors surfaced to a human
    plan_error: str | None           # last capability runtime failure, if any
    failed_capability: str | None    # name of the capability that failed
    failed_exc_type: str | None      # exception class name, for the diagnoser

    # Reflection / auto-fix control (Phase 5)
    reflection_history: list[dict[str, Any]]   # dumped ReflectionRecord per cycle
    reflection_attempts: dict[str, int]        # capability name -> reflect cycles used
    repair_attempts: int                       # total repairs applied this run
    last_repair: dict[str, Any] | None         # dumped RepairResult of most recent repair
    pending_retry_capability: str | None       # set by reflect_node so route re-dispatches
    escalate_to_planner: bool                  # reflect_node -> route -> "planner"
    planner_hint: dict[str, Any] | None        # dumped PlannerHint for the planner

    # Control
    status: str
    error: str | None
