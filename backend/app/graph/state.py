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
    training_report: dict[str, Any]  # per-model CV scores, best model path
    evaluation_report: dict[str, Any]  # held-out test metrics
    summary: str

    # Control
    status: str
    error: str | None
