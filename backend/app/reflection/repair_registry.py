"""Repair capability registry.

Mirrors ``app/planner/registry.py``: a frozen dataclass plus a name->capability
dict, so the mental model is identical to the planner's capability registry. A
repair capability declares which failure categories it handles and provides a
``runner`` with a fixed, total signature.

``REPAIR_REGISTRY`` holds ONLY artifact-level deterministic repairs — each fixes
a data artifact (or a documented spec/state knob) and lets the same capability
retry. Strategy changes (model swap, search-space reduction) are planner
concerns and live nowhere near this registry; they flow via
``Diagnosis.planner_hint``.

The registry is populated by ``app/reflection/repairs.py`` importing this module
and calling ``register`` for each capability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.graph.state import RunState
from app.reflection.models import Diagnosis, FailureCategory, RepairResult

# Runner contract: given the run state and a diagnosis, apply a deterministic
# repair and return a RepairResult. It MUST NOT raise for expected data
# conditions — it returns RepairResult(applied=False, error=...) instead, so the
# reflection node stays in control of the retry budget. It reads artifact paths
# from ``state``, rewrites the artifact on disk, and never mutates RunState or
# touches source code.
RepairRunner = Callable[[RunState, Diagnosis], RepairResult]


@dataclass(frozen=True)
class RepairCapability:
    name: str
    # Categories this repair is a candidate for. Used to map diagnosis -> repair.
    handles: frozenset[FailureCategory]
    runner: RepairRunner
    description: str = ""


REPAIR_REGISTRY: dict[str, RepairCapability] = {}


def register(cap: RepairCapability) -> None:
    """Register a repair capability. Later registrations override same-named
    ones, but names are expected to be unique."""
    REPAIR_REGISTRY[cap.name] = cap


def get(name: str) -> RepairCapability | None:
    return REPAIR_REGISTRY.get(name)


def candidates_for(category: FailureCategory) -> list[RepairCapability]:
    """Repairs that declare they handle this category, registration order
    preserved (so registration order == priority for category fallback)."""
    return [c for c in REPAIR_REGISTRY.values() if category in c.handles]
