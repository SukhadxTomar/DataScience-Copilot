"""Diagnosis -> repair-or-escalate selection.

The single decision point that enforces the architectural boundary: a diagnosis
either maps to an artifact-level repair (retry the same capability) or escalates
to the planner (change strategy). It never returns a planning action disguised as
a repair.

Selection is data-driven, not an if/else chain over categories:

1. If the diagnosis names a registered artifact repair -> use it.
2. Else if the diagnosis carries a PlannerHint (a planning concern) -> escalate
   with that hint, before any category fallback can shadow it.
3. Else if some registered repair handles the diagnosis category -> use the first
   (registration order == priority).
4. Else if the category is a known planning concern -> escalate with a
   synthesised hint.
5. Else -> escalate with no hint (the planner falls back to needs_input).

The result is a tagged ``RepairPlan``: exactly one of ``spec`` (retryable repair)
or ``escalate`` (hand to planner) is set, so the reflect node can branch cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.reflection.models import (
    Diagnosis,
    PlannerHint,
    PLANNING_CATEGORIES,
    RepairSpec,
)
from app.reflection.repair_registry import REPAIR_REGISTRY, candidates_for

# State key the artifact repairs target (see app/reflection/repairs.py). Carried
# on the RepairSpec so the node/audit trail records what the repair acts on.
_DEFAULT_ARTIFACT_KEY = "feature_report"


@dataclass(frozen=True)
class RepairPlan:
    """Outcome of selection. Exactly one of ``spec`` / ``escalate`` is meaningful:
    ``spec`` set -> run the repair and retry the capability; ``escalate`` True ->
    hand control to the planner, carrying ``planner_hint`` when present."""

    spec: RepairSpec | None = None
    escalate: bool = False
    planner_hint: PlannerHint | None = None


def _spec_for(name: str) -> RepairSpec:
    return RepairSpec(name=name, target_artifact_key=_DEFAULT_ARTIFACT_KEY)


def select_repair(diagnosis: Diagnosis, state: dict | None = None) -> RepairPlan:
    """Choose an artifact repair, or escalate to the planner.

    Never returns a planning action as a RepairSpec — model/config/resource
    concerns always escalate, carrying a PlannerHint when one is available.
    """
    # 1. An explicit, registered repair suggestion wins. The diagnoser sets this
    #    only for an artifact/knob fix (e.g. n_splits -> reduce_cv_folds), and
    #    never alongside a planner_hint, so it is unambiguous.
    suggested = diagnosis.suggested_repair
    if suggested and suggested in REPAIR_REGISTRY:
        return RepairPlan(spec=_spec_for(suggested))

    # 2. An explicit planner_hint is the diagnoser's signal that this is a
    #    PLANNING concern (swap the model, shrink the search space) — escalate
    #    before any category-repair fallback, so a repair that merely lists a
    #    planning category in its `handles` set can't shadow the hand-off.
    if diagnosis.planner_hint is not None:
        return RepairPlan(escalate=True, planner_hint=diagnosis.planner_hint)

    # 3. Otherwise, the first registered repair that handles this category.
    candidates = candidates_for(diagnosis.category)
    if candidates:
        return RepairPlan(spec=_spec_for(candidates[0].name))

    # 4./5. No artifact repair applies -> escalate to the planner. Synthesise a
    # default hint for a known planning concern, else escalate with no hint (the
    # planner falls back to needs_input).
    hint = None
    if diagnosis.category in PLANNING_CATEGORIES:
        hint = PlannerHint(note=diagnosis.reason)
    return RepairPlan(escalate=True, planner_hint=hint)
