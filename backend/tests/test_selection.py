"""Tests for select_repair — the diagnosis -> repair-or-escalate boundary (Phase C).

This is where the core architectural invariant is enforced: a diagnosis either
maps to a registered artifact repair (retry the same capability) or escalates to
the planner (change strategy). A planning concern is NEVER returned as a
RepairSpec.
"""

from __future__ import annotations

from app.reflection.models import (
    Diagnosis,
    FailureCategory,
    PlannerHint,
)
from app.reflection.selection import select_repair


def _diag(category, **kw) -> Diagnosis:
    kw.setdefault("confidence", 0.9)
    kw.setdefault("reason", "test")
    return Diagnosis(category=category, **kw)


def test_registered_suggested_repair_wins():
    d = _diag(
        FailureCategory.TARGET_ENCODING_ERROR,
        suggested_repair="label_encode_target",
    )
    plan = select_repair(d, state={})
    assert plan.escalate is False
    assert plan.spec is not None
    assert plan.spec.name == "label_encode_target"


def test_unknown_suggested_repair_falls_through_to_category():
    # A suggestion that is not in the registry is ignored; selection falls back
    # to the first registered repair that handles the category.
    d = _diag(
        FailureCategory.TARGET_ENCODING_ERROR,
        suggested_repair="not_a_real_repair",
    )
    plan = select_repair(d, state={})
    assert plan.spec is not None
    assert plan.spec.name == "label_encode_target"  # category handler


def test_category_candidate_used_when_no_suggestion():
    d = _diag(FailureCategory.MISSING_VALUES)
    plan = select_repair(d, state={})
    assert plan.spec is not None
    # impute_missing_values is registered first for MISSING_VALUES.
    assert plan.spec.name == "impute_missing_values"


def test_planning_category_escalates_with_synthesised_hint():
    # RESOURCE_LIMIT has no artifact repair -> escalate, and selection synthesises
    # a default hint from the diagnosis when none was attached.
    d = _diag(FailureCategory.RESOURCE_LIMIT, reason="OOM during fit")
    plan = select_repair(d, state={})
    assert plan.escalate is True
    assert plan.spec is None
    assert plan.planner_hint is not None
    assert plan.planner_hint.note == "OOM during fit"


def test_explicit_planner_hint_escalates_before_category_repair():
    # A diagnosis carrying a planner_hint is a PLANNING concern. Even though
    # MODEL_CONFIGURATION_ERROR also has registered artifact/knob repairs, the
    # explicit hint must force escalation — a data repair must never shadow the
    # planner hand-off. (The knob-repair path is the *suggested_repair* case,
    # which carries no hint; see test_registered_suggested_repair_wins.)
    hint = PlannerHint(requires_model_swap=True, note="swap the estimator")
    d = _diag(FailureCategory.MODEL_CONFIGURATION_ERROR, planner_hint=hint)
    plan = select_repair(d, state={})
    assert plan.escalate is True
    assert plan.spec is None
    assert plan.planner_hint is hint


def test_unknown_category_with_no_handler_escalates_without_hint():
    d = _diag(FailureCategory.UNKNOWN_ERROR, confidence=0.2)
    plan = select_repair(d, state={})
    assert plan.escalate is True
    assert plan.spec is None
    assert plan.planner_hint is None  # not a known planning concern


def test_model_config_hint_only_used_when_no_repair(monkeypatch):
    # If the only registered handler is removed, a MODEL_CONFIGURATION_ERROR must
    # escalate with a model-swap hint rather than inventing a repair.
    import app.reflection.selection as sel

    monkeypatch.setattr(sel, "candidates_for", lambda category: [])
    d = _diag(
        FailureCategory.MODEL_CONFIGURATION_ERROR,
        planner_hint=PlannerHint(requires_model_swap=True),
    )
    plan = select_repair(d, state={})
    assert plan.escalate is True
    assert plan.planner_hint is not None
    assert plan.planner_hint.requires_model_swap is True
