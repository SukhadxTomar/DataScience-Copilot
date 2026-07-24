"""Capability registry.

A capability is a plannable pipeline step: a name, the state keys it
``requires`` before it can run, the state keys it ``produces``, and a
``runner`` that does the work. The runners are the *existing* graph node
functions unchanged — the registry is a thin metadata layer that lets a
planner reason about, and a validator check, the requires/produces DAG
without touching the node bodies.

``profile`` is deliberately not a capability: it is the bootstrap step that
runs before planning (the planner needs the profile to plan), so it is
never something the planner schedules.

Adding a new step to the pipeline is now a matter of writing a node
function and registering it here — no graph rewiring required.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.graph.nodes import (
    cleaning_node,
    evaluation_node,
    explain_node,
    feature_plan_node,
    features_node,
    insights_node,
    problem_spec_node,
    recommendations_node,
    report_node,
    summarize_node,
    training_node,
)
from app.graph.state import RunState

# State keys always present once the bootstrap profile step has run. The
# planner and validator treat these as already produced.
BOOTSTRAP_ARTIFACTS: frozenset[str] = frozenset(
    {"run_id", "dataset_id", "problem_text", "profile"}
)

# Terminal artifacts a valid plan must produce for the run to be complete.
# ``report`` transitively forces the whole modeling chain via its requires;
# ``summary`` stays terminal so ``summarize`` always runs (the API/report use it).
TERMINAL_ARTIFACTS: frozenset[str] = frozenset({"summary", "report"})


@dataclass(frozen=True)
class Capability:
    name: str
    requires: frozenset[str]
    produces: frozenset[str]
    needs_llm: bool
    runner: Callable[[RunState], dict]


# Note: ``insights`` is intentionally NOT in ``summarize``'s requires — it is
# the one capability nothing hard-depends on, so the planner may legitimately
# include it or skip it. ``summarize_node`` degrades gracefully without it.
_CAPABILITIES: list[Capability] = [
    Capability(
        name="insights",
        requires=frozenset({"profile"}),
        produces=frozenset({"insights"}),
        needs_llm=True,
        runner=insights_node,
    ),
    Capability(
        name="problem_spec",
        requires=frozenset({"problem_text", "profile"}),
        produces=frozenset({"problem_spec"}),
        needs_llm=True,
        runner=problem_spec_node,
    ),
    Capability(
        name="cleaning",
        requires=frozenset({"problem_spec"}),
        produces=frozenset({"cleaning_report"}),
        needs_llm=False,
        runner=cleaning_node,
    ),
    Capability(
        name="feature_plan",
        requires=frozenset({"cleaning_report", "problem_spec"}),
        produces=frozenset({"feature_plan"}),
        needs_llm=True,
        runner=feature_plan_node,
    ),
    Capability(
        name="features",
        requires=frozenset({"feature_plan", "problem_spec", "cleaning_report"}),
        produces=frozenset({"feature_report"}),
        needs_llm=False,
        runner=features_node,
    ),
    Capability(
        name="training",
        requires=frozenset({"feature_report", "problem_spec"}),
        produces=frozenset({"training_report"}),
        needs_llm=False,
        runner=training_node,
    ),
    Capability(
        name="evaluation",
        requires=frozenset({"feature_report", "training_report", "problem_spec"}),
        produces=frozenset({"evaluation_report"}),
        needs_llm=False,
        runner=evaluation_node,
    ),
    Capability(
        name="explain",
        requires=frozenset({"training_report", "feature_report", "problem_spec"}),
        produces=frozenset({"explanation_report"}),
        needs_llm=False,
        runner=explain_node,
    ),
    Capability(
        name="recommendations",
        requires=frozenset({"problem_spec", "evaluation_report", "explanation_report"}),
        produces=frozenset({"recommendations"}),
        needs_llm=True,
        runner=recommendations_node,
    ),
    Capability(
        # insights/summary deliberately NOT required (both optional), so a
        # skipped or late-running one never blocks the report.
        name="report",
        requires=frozenset(
            {
                "profile",
                "problem_spec",
                "cleaning_report",
                "feature_report",
                "training_report",
                "evaluation_report",
                "explanation_report",
                "recommendations",
            }
        ),
        produces=frozenset({"report"}),
        needs_llm=False,
        runner=report_node,
    ),
    Capability(
        name="summarize",
        requires=frozenset(
            {
                "problem_spec",
                "cleaning_report",
                "feature_report",
                "training_report",
                "evaluation_report",
            }
        ),
        # ``status`` is control state (it already exists as "running" in the
        # initial state), so it is not a produced *data* artifact — only
        # ``summary`` is, and it is the plan's terminal.
        produces=frozenset({"summary"}),
        needs_llm=False,
        runner=summarize_node,
    ),
]

REGISTRY: dict[str, Capability] = {cap.name: cap for cap in _CAPABILITIES}

# The full canonical pipeline order: the historical linear sequence and a
# known validator-passing plan. Used as a reference and in tests.
CANONICAL_PLAN: list[str] = [
    "insights",
    "problem_spec",
    "cleaning",
    "feature_plan",
    "features",
    "training",
    "evaluation",
    "explain",
    "recommendations",
    "summarize",
    "report",
]


def get(name: str) -> Capability:
    return REGISTRY[name]


def all_names() -> list[str]:
    return list(REGISTRY.keys())


def describe_for_prompt() -> str:
    """Render the registry as a compact capability list for the planner prompt."""
    lines: list[str] = []
    for cap in REGISTRY.values():
        requires = ", ".join(sorted(cap.requires)) or "nothing"
        produces = ", ".join(sorted(cap.produces))
        lines.append(f"- {cap.name} — requires: {requires}; produces: {produces}")
    return "\n".join(lines)
