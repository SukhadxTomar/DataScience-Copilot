"""Tests for reflect_node — the diagnose -> repair -> retry policy (Phase D).

reflect_node is the only place the auto-fix control policy lives. These tests
drive it directly with hand-built state (no graph) and assert the partial state
update it returns: repair-and-retry on success, escalation on budget exhaustion /
no-op / planning concern, and knob persistence into problem_spec.

All failures used here are confident heuristic hits, so the diagnoser never
touches the LLM; the patched client is a guard that fails loudly if it is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings
from app.reflection import node as reflect_mod
from app.reflection.node import reflect_node


class _NoLLM:
    """Stand-in LLM client that must never be called (heuristic-only paths)."""

    def complete(self, *, system, user, schema):  # pragma: no cover - guard
        raise AssertionError("LLM must not be consulted for a confident heuristic")


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    monkeypatch.setattr(reflect_mod, "get_llm_client", lambda: _NoLLM())


def _features(tmp_path: Path, df: pd.DataFrame) -> str:
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return str(path)


# --- happy path: repair applied, capability re-dispatched -------------------


def test_target_encoding_failure_repairs_and_retries(tmp_path):
    fpath = _features(tmp_path, pd.DataFrame({"x": [1.0, 2.0], "y": ["No", "Yes"]}))
    state = {
        "failed_capability": "training",
        "plan_error": "could not convert string to float: 'Yes'",
        "failed_exc_type": "ValueError",
        "problem_spec": {"target_column": "y"},
        "feature_report": {"features_path": fpath},
    }

    out = reflect_node(state)

    # The same capability is parked for re-dispatch; the failure is cleared.
    assert out["pending_retry_capability"] == "training"
    assert out["plan_error"] is None
    assert out["failed_capability"] is None
    assert out["escalate_to_planner"] is False
    # Budgets advanced and the cycle recorded.
    assert out["repair_attempts"] == 1
    assert out["reflection_attempts"]["training"] == 1
    assert out["last_repair"]["repair_name"] == "label_encode_target"
    assert out["reflection_history"][-1]["diagnosis"]["category"] == "TARGET_ENCODING_ERROR"
    # The artifact really was rewritten in place.
    assert pd.read_parquet(fpath)["y"].tolist() == [0, 1]


# --- budgets ----------------------------------------------------------------


def test_reflection_budget_exhaustion_escalates(tmp_path):
    fpath = _features(tmp_path, pd.DataFrame({"x": [1.0], "y": ["No"]}))
    state = {
        "failed_capability": "training",
        "plan_error": "could not convert string to float: 'Yes'",
        "failed_exc_type": "ValueError",
        "problem_spec": {"target_column": "y"},
        "feature_report": {"features_path": fpath},
        # Already used the per-capability budget.
        "reflection_attempts": {"training": settings.reflection_attempts},
    }

    out = reflect_node(state)

    assert out["escalate_to_planner"] is True
    assert out["pending_retry_capability"] is None
    assert any("exhausted" in w for w in out["warnings"])


def test_repair_budget_exhaustion_escalates(tmp_path):
    fpath = _features(tmp_path, pd.DataFrame({"x": [1.0], "y": ["No"]}))
    state = {
        "failed_capability": "training",
        "plan_error": "could not convert string to float: 'Yes'",
        "failed_exc_type": "ValueError",
        "problem_spec": {"target_column": "y"},
        "feature_report": {"features_path": fpath},
        "repair_attempts": settings.repair_attempts,  # run-wide budget spent
    }

    out = reflect_node(state)

    assert out["escalate_to_planner"] is True
    assert any("budget" in w for w in out["warnings"])


def test_missing_failed_capability_escalates_defensively():
    out = reflect_node({"plan_error": "boom"})
    assert out["escalate_to_planner"] is True
    assert out["pending_retry_capability"] is None


# --- no-op repair must not burn a retry -------------------------------------


def test_noop_repair_escalates_without_retry(tmp_path):
    # Target already numeric -> label_encode_target is a no-op (changed=False).
    # The node must escalate rather than re-dispatch an unchanged artifact.
    fpath = _features(tmp_path, pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}))
    state = {
        "failed_capability": "training",
        "plan_error": "could not convert string to float: 'Yes'",
        "failed_exc_type": "ValueError",
        "problem_spec": {"target_column": "y"},
        "feature_report": {"features_path": fpath},
    }

    out = reflect_node(state)

    assert out["escalate_to_planner"] is True
    assert out.get("pending_retry_capability") is None
    # The attempt is still counted (it consumed a diagnosis + repair try).
    assert out["reflection_attempts"]["training"] == 1


# --- knob repair persists into problem_spec ---------------------------------


def test_cv_folds_knob_repair_updates_problem_spec():
    # A MODEL_CONFIGURATION_ERROR about folds maps to the reduce_cv_folds knob,
    # which the node must materialise into problem_spec (no dataset rewrite).
    state = {
        "failed_capability": "training",
        "plan_error": "n_splits=5 cannot be greater than the number of members in each class.",
        "failed_exc_type": "ValueError",
        "problem_spec": {"target_column": "y", "cv_folds": 5},
    }

    out = reflect_node(state)

    assert out["pending_retry_capability"] == "training"
    assert out["problem_spec"]["cv_folds"] == 4  # 5 -> 4
    # Original spec object is not mutated in place (deep-copied).
    assert state["problem_spec"]["cv_folds"] == 5
    assert out["last_repair"]["artifact_key"] == "cv_folds"


# --- planning concern escalates with a hint ---------------------------------


def test_resource_limit_escalates_with_planner_hint():
    state = {
        "failed_capability": "training",
        "plan_error": "Unable to allocate 4.5 GiB for an array",
        "failed_exc_type": "MemoryError",
        "problem_spec": {"target_column": "y"},
    }

    out = reflect_node(state)

    assert out["escalate_to_planner"] is True
    assert out["pending_retry_capability"] is None
    assert out["planner_hint"] is not None
    assert out["planner_hint"]["reduce_search_space"] is True
    # The diagnosis was still recorded for the audit trail.
    assert out["reflection_history"][-1]["diagnosis"]["category"] == "RESOURCE_LIMIT"
