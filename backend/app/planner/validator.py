"""Execution-plan validator.

Pure, side-effect-free check that an ordered list of capability names forms
a runnable DAG given what is already available. This is the safety net that
lets an LLM propose a plan: whatever the model returns, nothing executes
until the plan is proven sound against the registry's requires/produces.

The error strings are written to be human- and LLM-readable — they are fed
straight back to the planner as feedback on a retry.
"""

from __future__ import annotations

from app.planner.registry import (
    TERMINAL_ARTIFACTS,
    Capability,
)


def validate_plan(
    steps: list[str],
    initial_artifacts: set[str],
    registry: dict[str, Capability],
) -> tuple[bool, list[str]]:
    """Validate an ordered plan.

    Checks, in order:
      (a) every step names a registered capability;
      (b) walking the plan, each step's ``requires`` are already produced;
      (c) no artifact is produced by two different steps;
      (d) the plan ends having produced the terminal artifacts.

    Returns ``(ok, errors)`` where ``ok`` is ``True`` only if ``errors`` is
    empty.
    """
    errors: list[str] = []

    # (a) All names registered. If any are unknown we cannot safely walk the
    # plan (no requires/produces to reason about), so report and stop.
    unknown = [name for name in steps if name not in registry]
    if unknown:
        for name in unknown:
            errors.append(f"unknown capability '{name}' is not in the registry")
        return False, errors

    if not steps:
        errors.append("plan is empty")
        return False, errors

    produced: set[str] = set(initial_artifacts)
    producers: dict[str, str] = {key: "<initial>" for key in initial_artifacts}

    for i, name in enumerate(steps):
        cap = registry[name]

        # (b) requires must already be satisfied at this point in the order.
        missing = cap.requires - produced
        if missing:
            errors.append(
                f"step {i} '{name}' needs {sorted(missing)} which "
                f"no earlier step produces"
            )

        # (c) no duplicate producers.
        for key in cap.produces:
            if key in producers:
                errors.append(
                    f"step {i} '{name}' re-produces '{key}' already produced "
                    f"by '{producers[key]}'"
                )
            producers[key] = name

        produced |= cap.produces

    # (d) the plan must terminate by producing the run's terminal artifacts.
    for terminal in sorted(TERMINAL_ARTIFACTS):
        if terminal not in produced:
            errors.append(f"plan never produces the required '{terminal}'")

    return (not errors), errors
