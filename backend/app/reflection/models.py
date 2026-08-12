"""Structured models for the reflection + auto-fix layer.

Every payload the reflection layer produces is a Pydantic model, matching the
convention used by the agents and the planner (``ExecutionPlan``, ``ProblemSpec``,
the report models). Models are stored into ``RunState`` via ``.model_dump()`` and
rehydrated on read, so they stay small and JSON-serialisable for SQLite
checkpointing.

The core architectural boundary lives here: a *repair* fixes a data artifact and
lets the same capability retry (``RepairSpec`` / ``RepairResult``); a *planner
hint* (``PlannerHint``) asks the planner to change execution strategy. These are
different actions and are never conflated — the reflection layer never rewrites
the plan itself.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """Fixed taxonomy. Every diagnosis resolves to exactly one member."""

    DATA_SCHEMA_ERROR = "DATA_SCHEMA_ERROR"
    TARGET_ENCODING_ERROR = "TARGET_ENCODING_ERROR"
    MISSING_VALUES = "MISSING_VALUES"
    TYPE_ERROR = "TYPE_ERROR"
    MODEL_CONFIGURATION_ERROR = "MODEL_CONFIGURATION_ERROR"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    TRANSIENT_ERROR = "TRANSIENT_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# Categories that are planning concerns, not artifact defects. A diagnosis in one
# of these escalates to the planner (carrying a PlannerHint) rather than mapping
# to an artifact repair. Kept here so selection logic and tests share one source.
PLANNING_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {FailureCategory.MODEL_CONFIGURATION_ERROR, FailureCategory.RESOURCE_LIMIT}
)


class PlannerHint(BaseModel):
    """Advice the reflection layer hands to the planner when no artifact-level
    repair applies.

    This is the ONLY channel by which a diagnosis influences planning — the
    reflection layer never rewrites the plan itself. A repair fixes an artifact
    and lets the same capability retry; a planner hint asks the planner to change
    execution *strategy* (swap the model, shrink the search space, pick a
    different capability, or abort).
    """

    requires_model_swap: bool = False  # e.g. MODEL_CONFIGURATION_ERROR
    reduce_search_space: bool = False  # e.g. RESOURCE_LIMIT (fewer models/folds)
    note: str = Field(
        default="",
        description="Free-text guidance for the planner, e.g. 'xgboost OOM; "
        "prefer a lighter estimator'",
    )


class Diagnosis(BaseModel):
    """The reflection agent's structured verdict on one failure."""

    category: FailureCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Plain-language root cause, 1-2 sentences")
    suggested_repair: str | None = Field(
        default=None,
        description="Name of an ARTIFACT-LEVEL repair from REPAIR_REGISTRY, or "
        "None. Never a planning action — those are carried by planner_hint.",
    )
    # Set when the failure is a planning concern (model/config/resource), not an
    # artifact defect. Selection escalates to the planner with this hint instead
    # of returning a RepairSpec. None when an artifact repair applies.
    planner_hint: PlannerHint | None = None
    # Free-form hints the diagnoser extracted (e.g. offending column, bad dtype).
    # Repairs read these instead of re-parsing the exception string.
    evidence: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(
        default="heuristic",
        description="'heuristic' | 'llm' — how the diagnosis was reached",
    )


class RepairSpec(BaseModel):
    """A concrete artifact repair to attempt: which capability, against which
    artifact, with which parameters. Built by mapping a Diagnosis onto the
    repair registry."""

    name: str  # key in REPAIR_REGISTRY
    target_artifact_key: str  # e.g. "feature_report" — state key holding the path
    params: dict[str, Any] = Field(default_factory=dict)


class RepairResult(BaseModel):
    """Outcome of one repair attempt. Deterministic, serialisable, auditable."""

    repair_name: str
    applied: bool  # did the repair run to completion?
    changed: bool  # did it actually modify an artifact?
    detail: str  # human summary ("encoded target Yes/No -> 1/0")
    artifact_key: str | None = None  # state key whose artifact was rewritten
    new_artifact_path: str | None = None  # if the repair wrote a new path
    error: str | None = None  # populated iff the repair itself failed


class ReflectionRecord(BaseModel):
    """One full diagnose -> repair -> retry cycle, persisted per run for audit
    and for the planner to consume on escalation."""

    failed_capability: str
    original_error: str
    diagnosis: Diagnosis
    repair: RepairSpec | None = None  # None if no artifact repair mapped
    repair_result: RepairResult | None = None
    retry_succeeded: bool | None = None  # None until the capability re-runs
    attempt: int  # 1-based reflection attempt for this capability
    duration_ms: int
