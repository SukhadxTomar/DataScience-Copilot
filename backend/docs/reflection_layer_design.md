# Reflection + Auto-Fix Layer — Design

Evolving the executor from *blind retry → needs_input* into a **self-healing loop**:
diagnose the failure, apply a deterministic repair to the on-disk artifact, re-run the
same capability, and only escalate to the planner once the repair budget is exhausted.

This design is written against the existing code and reuses its invariants rather than
replacing them:

- **State carries paths and reports, never dataframes or models** (`graph/state.py`).
  Repairs therefore operate on *artifacts on disk* and record what they touched in
  small state entries — same contract every tool already honours.
- **The capability wrapper already traps exceptions** into `plan_error` /
  `failed_capability` without advancing the cursor (`graph/builder.py::_make_capability_node`).
  That is exactly the hook the reflection node needs — no change to how failures are surfaced.
- **`route` is the single conditional edge** every node returns through
  (`planner/planner_node.py::route`). The new node sits on that seam: on a failure `route`
  sends control to `reflect` instead of straight to `planner`.
- **Repairs are registry-based**, mirroring the existing `Capability` registry, so the
  executor never grows an `if/else` chain and adding a repair is a one-line registration.

---

## 1. Folder / module structure

New package `app/reflection/`, plus one new tools module for the actual repair mechanics.
Nothing existing is deleted; the changes to existing files are additive and small.

```
backend/app/
├── reflection/
│   ├── __init__.py
│   ├── models.py            # Pydantic: FailureCategory, Diagnosis, RepairSpec,
│   │                        #           RepairResult, ReflectionRecord
│   ├── taxonomy.py          # exception → FailureCategory heuristics (deterministic
│   │                        #           first pass, before any LLM call)
│   ├── diagnoser.py         # ReflectionAgent: heuristic + optional LLM diagnosis
│   ├── repair_registry.py   # RepairCapability dataclass + REPAIR_REGISTRY, mirrors
│   │                        #           planner/registry.py
│   ├── repairs.py           # the repair capability functions (thin: parse + delegate)
│   └── node.py              # reflect_node + route wiring helpers (LangGraph node)
├── tools/
│   └── repair_ops.py        # deterministic dataframe/artifact repair primitives
│                            #   (label_encode_target, coerce_numeric, impute, …)
├── graph/
│   ├── state.py             # + reflection control keys (edit)
│   └── builder.py           # register the reflect node + route it (edit)
├── planner/
│   └── planner_node.py      # route(): failure → "reflect"; planner consumes
│                            #           ReflectionRecord history on escalation (edit)
└── core/
    └── config.py            # + reflection_attempts / repair_attempts budgets (edit)
```

**Layering (Clean Architecture).** Dependencies point inward only:

```
graph/node (reflect_node)  ─┐
                            ├─▶ reflection/diagnoser  ─▶ llm/base (Protocol)
reflection/repair_registry ─┘         │
        │                             ▼
        ▼                        reflection/models (Pydantic — no deps)
reflection/repairs ─▶ tools/repair_ops ─▶ pandas / sklearn
```

- `reflection/models.py` depends on nothing but Pydantic — the stable core.
- `tools/repair_ops.py` is pure dataframe work (no state, no LLM, no LangGraph) — unit-testable
  in isolation, exactly like `tools/cleaning.py` today.
- `reflection/diagnoser.py` depends only on the `LLMClient` **Protocol**, never a concrete
  provider — same rule the other agents follow.
- `reflection/node.py` is the only reflection module that knows about `RunState`.

---

## 2. Core Pydantic models  (`app/reflection/models.py`)

The rest of the state layer is `TypedDict`, but structured *payloads* in this codebase are
Pydantic (`ExecutionPlan`, agent reports) and stored into state via `.model_dump()`. The
reflection payloads follow that same pattern.

```python
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


class PlannerHint(BaseModel):
    """Advice the reflection layer hands to the planner when no artifact-level
    repair applies. This is the ONLY channel by which a diagnosis influences
    planning — the reflection layer never rewrites the plan itself.

    A repair fixes an artifact and lets the same capability retry. A planner
    hint, by contrast, asks the planner to change execution *strategy* (swap the
    model, shrink the search space, pick a different capability, or abort). These
    are categorically different actions and are kept in separate types."""
    requires_model_swap: bool = False      # e.g. MODEL_CONFIGURATION_ERROR
    reduce_search_space: bool = False      # e.g. RESOURCE_LIMIT (fewer models/folds)
    note: str = Field(
        default="",
        description="Free-text guidance for the planner, e.g. 'xgboost OOM; "
        "prefer a lighter estimator'",
    )


class Diagnosis(BaseModel):
    """The reflection agent's structured verdict on one failure."""
    category: FailureCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Plain-language root cause, 1–2 sentences")
    suggested_repair: str | None = Field(
        default=None,
        description="Name of an ARTIFACT-LEVEL repair from REPAIR_REGISTRY, or "
        "None. Never a planning action — those are carried by planner_hint.",
    )
    # Set when the failure is a planning concern (model/config/resource), not an
    # artifact defect. select_repair() escalates to the planner with this hint
    # instead of returning a RepairSpec. None when an artifact repair applies.
    planner_hint: PlannerHint | None = Field(default=None)
    # Free-form hints the diagnoser extracted (e.g. offending column, bad dtype).
    # Repairs read these instead of re-parsing the exception string.
    evidence: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(
        default="heuristic",
        description="'heuristic' | 'llm' — how the diagnosis was reached",
    )


class RepairSpec(BaseModel):
    """A concrete repair to attempt: which capability, against which artifact,
    with which parameters. Built by mapping a Diagnosis onto the registry."""
    name: str                              # key in REPAIR_REGISTRY
    target_artifact_key: str               # e.g. "feature_report" — where the path lives
    params: dict[str, Any] = Field(default_factory=dict)


class RepairResult(BaseModel):
    """Outcome of one repair attempt. Deterministic, serialisable, auditable."""
    repair_name: str
    applied: bool                          # did the repair run to completion?
    changed: bool                          # did it actually modify an artifact?
    detail: str                            # human summary ("encoded target Yes/No -> 1/0")
    artifact_key: str | None = None        # state key whose artifact was rewritten
    new_artifact_path: str | None = None   # if the repair wrote a new path
    error: str | None = None               # populated iff the repair itself failed


class ReflectionRecord(BaseModel):
    """One full diagnose→repair→retry cycle, persisted per run for audit and
    for the planner to consume on escalation."""
    failed_capability: str
    original_error: str
    diagnosis: Diagnosis
    repair: RepairSpec | None              # None if no repair mapped
    repair_result: RepairResult | None
    retry_succeeded: bool | None           # None until the capability re-runs
    attempt: int                           # 1-based reflection attempt for this capability
    duration_ms: int
```

### State updates (`app/graph/state.py`)

Add reflection control keys alongside the existing planning keys. All small and serialisable —
the checkpoint stays cheap. Repair *history* is a list of dumped `ReflectionRecord`s.

```python
class RunState(TypedDict, total=False):
    # … existing keys unchanged …

    # Reflection / auto-fix control
    reflection_history: list[dict[str, Any]]   # dumped ReflectionRecord per cycle
    reflection_attempts: dict[str, int]        # capability name -> reflect cycles used
    repair_attempts: int                       # total repairs applied this run
    last_repair: dict[str, Any] | None         # dumped RepairResult of most recent repair
    pending_retry_capability: str | None       # set by reflect_node so route re-dispatches it
    failed_exc_type: str | None                # exception class name, for the diagnoser
    escalate_to_planner: bool                  # reflect_node -> route -> "planner"
    planner_hint: dict[str, Any] | None        # dumped PlannerHint for the planner to consume
```

**Why `reflection_attempts` is a per-capability dict:** the budget is enforced *per failing
step*, not globally — training failing twice shouldn't consume the budget that evaluation
might later need. `repair_attempts` (a flat counter) is the run-wide backstop.

---

## 3. Interfaces

### 3.1 Repair capability contract  (`app/reflection/repair_registry.py`)

Mirrors `planner/registry.py::Capability` exactly, so the mental model is identical.
A repair is: a name, the failure categories it addresses, the artifact it targets, and a
pure runner.

```python
from collections.abc import Callable
from dataclasses import dataclass, field

from app.graph.state import RunState
from app.reflection.models import Diagnosis, RepairResult


@dataclass(frozen=True)
class RepairCapability:
    name: str
    # Categories this repair is a candidate for. Used to map diagnosis -> repair.
    handles: frozenset[FailureCategory]
    # Runner signature is fixed and total: (state, diagnosis) -> RepairResult.
    # It MUST NOT raise for expected data conditions — it returns
    # RepairResult(applied=False, error=...) instead, so the node stays in control
    # of the budget. It reads artifact paths from `state`, rewrites the artifact on
    # disk, and returns what changed. It never touches source code and never mutates
    # RunState directly.
    runner: Callable[[RunState, Diagnosis], RepairResult]
    description: str = ""


REPAIR_REGISTRY: dict[str, RepairCapability] = {}


def register(cap: RepairCapability) -> None:
    REPAIR_REGISTRY[cap.name] = cap


def candidates_for(category: FailureCategory) -> list[RepairCapability]:
    """Repairs that declare they handle this category, registration order preserved."""
    return [c for c in REPAIR_REGISTRY.values() if category in c.handles]
```

**Repair capability catalogue** (each is one `RepairCapability` registration; runner bodies
live in `repairs.py`, mechanics in `tools/repair_ops.py`):

| repair | handles | what it does to the artifact |
|---|---|---|
| `label_encode_target` | TARGET_ENCODING_ERROR | map non-numeric target (`Yes/No`) → `0/1`, persist mapping in `problem_spec` |
| `coerce_numeric` | TYPE_ERROR | `pd.to_numeric(errors="coerce")` on offending columns |
| `convert_datetime_columns` | TYPE_ERROR, DATA_SCHEMA_ERROR | parse object columns that are dates; expand or drop |
| `impute_missing_values` | MISSING_VALUES | fill NaNs (median/most-frequent) in the features artifact |
| `remove_invalid_rows` | MISSING_VALUES, VALIDATION_ERROR | drop rows with NaN target / inf values |
| `drop_constant_columns` | DATA_SCHEMA_ERROR, MODEL_CONFIGURATION_ERROR | drop zero-variance / all-NaN columns |
| `validate_schema` | DATA_SCHEMA_ERROR | realign feature columns to the model's expected set |
| `reduce_cv_folds` | MODEL_CONFIGURATION_ERROR | lower `CV_FOLDS` when a class has too few samples |
| `retry_llm` | TRANSIENT_ERROR | no artifact change; signal a plain re-run (backoff) |

**`REPAIR_REGISTRY` holds only artifact-level deterministic repairs.** Every entry above fixes
a data artifact (or a documented spec/state knob) and lets the *same* capability retry
immediately, returning a `RepairResult`. Changing execution *strategy* — swapping the model,
shrinking the model search space, choosing a different capability, aborting — is a **planning**
decision, not a repair. It is carried by `Diagnosis.planner_hint` (§2) and actioned by the
planner (§4), never registered as a repair.

> `reduce_cv_folds` needs a small parameter surface on the training tool. To keep repairs from
> editing source, expose it as a **state/spec knob** the tool reads (see §5, migration):
> `train_models` gains an optional `cv_folds` argument, defaulted so current behaviour is
> unchanged. The repair writes the knob into `problem_spec`/state; the tool honours it. This
> preserves "repairs never touch source." Model-swap and search-space reduction are deliberately
> NOT repairs — they flow to the planner via `planner_hint`, which owns the plan.

### 3.2 ReflectionAgent (diagnoser)  (`app/reflection/diagnoser.py`)

```python
class ReflectionAgent:
    """Diagnoses a capability failure into a structured Diagnosis.

    Two-tier by design and by cost:
      1. Deterministic heuristics (taxonomy.py) classify the exception by type
         and message pattern. Fast, free, and covers the known failure modes
         (the XGBoost target-encoding case is a pure heuristic hit).
      2. Only when heuristics are low-confidence does it fall back to the LLM,
         which must return a schema-valid Diagnosis (reuses the LLMClient
         Protocol + structured-output contract every other agent uses).

    The LLM fallback runs under its own timeout (``diagnosis_timeout_s``). A
    hanging provider must never stall the graph: on timeout (or any LLMError)
    the agent returns the low-confidence heuristic Diagnosis it already has,
    marked ``source='heuristic'``. Diagnosis is best-effort — a slow provider
    degrades to the heuristic verdict, it does not block the run.
    """
    def __init__(
        self,
        llm: LLMClient,
        min_confidence: float = 0.6,
        diagnosis_timeout_s: float = settings.diagnosis_timeout_s,
    ) -> None: ...

    def diagnose(
        self,
        *,
        failed_capability: str,
        error: str,
        exc_type: str,
        state: RunState,
    ) -> Diagnosis:
        ...
```

`taxonomy.py` is a pure function `classify(exc_type: str, message: str) -> Diagnosis` — an
ordered list of `(predicate, category, confidence, evidence-extractor)` rules. Deterministic
and unit-testable with plain strings; no LLM, no state.

**Unknown messages fall back to the LLM, they do not silently become `UNKNOWN_ERROR`.**
`classify()` returns a low-confidence (`< min_confidence`) `UNKNOWN_ERROR` diagnosis when no
rule matches. Because it is below threshold, the agent treats it as "heuristics couldn't
decide" and invokes the LLM fallback. `UNKNOWN_ERROR` is thus a *routing signal to the LLM*,
not a terminal verdict — only surfaced as final if the LLM is unavailable or also can't
classify. Tests assert an unmatched message triggers exactly one LLM call.

### 3.3 RepairRegistry (mapping diagnosis → repair or planner hint)

The mapping is data-driven, not an `if/else` chain, and returns a **tagged outcome** so the
node can tell a retryable repair apart from a planning escalation:

1. If `diagnosis.suggested_repair` names a registered artifact repair → return a `RepairSpec`.
2. Else if `candidates_for(diagnosis.category)` is non-empty → return the first (registration
   order = priority) as a `RepairSpec`.
3. Else if `diagnosis.planner_hint` is set (or the category is a planning concern —
   `MODEL_CONFIGURATION_ERROR` / `RESOURCE_LIMIT`) → escalate to the planner with the hint.
4. Else → escalate to the planner with the diagnosis attached (no hint).

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RepairPlan:
    """Result of selection: exactly one of `spec` (retryable artifact repair) or
    `escalate` (hand to planner) is set."""
    spec: RepairSpec | None = None
    escalate: bool = False
    planner_hint: PlannerHint | None = None


def select_repair(diagnosis: Diagnosis, state: RunState) -> RepairPlan:
    """Choose an artifact repair, or escalate to the planner.

    Never returns a planning action as a RepairSpec — model/config/resource
    concerns always escalate carrying a PlannerHint.
    """
    ...
```

`reflect_node` (§4) branches on the returned `RepairPlan`: `spec` set → run the repair and
retry the same capability; `escalate` set → set `escalate_to_planner` + stash the
`planner_hint` in state for the planner to consume.

---

## 4. LangGraph node/edge changes

### The new node  (`app/reflection/node.py`)

`reflect_node` is a normal node — reads state, does work, returns a partial update. It is the
only place the diagnose→repair→retry policy lives.

```python
def reflect_node(state: RunState) -> dict:
    failed = state["failed_capability"]
    error = state["plan_error"]
    attempts = dict(state.get("reflection_attempts", {}))
    used = attempts.get(failed, 0)

    # Budget check first — exhausted reflection escalates to the planner with history.
    if used >= settings.reflection_attempts or \
       state.get("repair_attempts", 0) >= settings.repair_attempts:
        return _escalate(state, reason="reflection budget exhausted")

    t0 = _monotonic_ms()
    diagnosis = ReflectionAgent(get_llm_client()).diagnose(
        failed_capability=failed, error=error,
        exc_type=state.get("failed_exc_type", ""), state=state,
    )
    plan = select_repair(diagnosis, state)

    # Planning concern (model/config/resource) — no artifact repair applies.
    # Escalate to the planner carrying the hint; the planner owns the plan.
    if plan.escalate:
        record = _record(failed, error, diagnosis, None, None, used + 1, t0)
        return _escalate(state, reason="planner escalation", record=record,
                         planner_hint=plan.planner_hint)

    spec = plan.spec
    result = REPAIR_REGISTRY[spec.name].runner(state, diagnosis)
    record = _record(failed, error, diagnosis, spec, result, used + 1, t0)
    attempts[failed] = used + 1

    if not result.applied or not result.changed:
        # Repair couldn't help — don't burn a retry on an unchanged artifact.
        return _escalate(state, reason="repair made no change", record=record,
                         attempts=attempts)

    # Repair succeeded: clear the error and re-dispatch the SAME capability.
    return {
        "plan_error": None,
        "failed_capability": None,
        "failed_exc_type": None,
        "pending_retry_capability": failed,     # route() reads this
        "reflection_attempts": attempts,
        "repair_attempts": state.get("repair_attempts", 0) + 1,
        "last_repair": result.model_dump(),
        "reflection_history": state.get("reflection_history", []) + [record.model_dump()],
    }
```

Key point: **the cursor is never advanced by a failure or a repair.** `plan_cursor` still
points at the failed step. Clearing `plan_error` + setting `pending_retry_capability` makes
`route` send control back to that same capability node — the retry is automatic and needs no
change to the capability wrapper.

### `route` changes  (`app/planner/planner_node.py`)

Currently: `plan_error` set → return `"planner"`. New: → return `"reflect"`.
`reflect_node` decides whether the outcome is a retry (re-dispatch the capability) or an
escalation (fall through to `"planner"`).

```python
def route(state: RunState) -> str:
    if state.get("status") == "needs_input":
        return "__end__"
    if state.get("plan_error"):
        return "reflect"                       # was "planner"
    if state.get("escalate_to_planner"):       # set by reflect_node on exhaustion
        return "planner"
    if cap := state.get("pending_retry_capability"):
        return cap                             # re-run the just-repaired capability
    plan = state.get("execution_plan") or []
    cursor = state.get("plan_cursor", 0)
    return "__end__" if cursor >= len(plan) else plan[cursor]
```

The capability wrapper (`_make_capability_node`) also clears `pending_retry_capability` on
entry so a *successful* retry doesn't loop. One extra line:

```python
def _node(state: RunState) -> dict:
    try:
        produced = cap.runner(state)
    except Exception as exc:
        return {"plan_error": str(exc), "failed_capability": cap.name,
                "failed_exc_type": type(exc).__name__,        # new: help the diagnoser
                "pending_retry_capability": None}
    cursor = state.get("plan_cursor", 0)
    return {**produced, "plan_cursor": cursor + 1, "pending_retry_capability": None}
```

### `builder.py` changes

Register one node and add it to the route map. The reflect node routes back through the same
`route` function, so it participates in the existing walk.

```python
from app.reflection.node import reflect_node

graph.add_node("reflect", reflect_node)

path_map = (
    {name: name for name in REGISTRY}
    | {"planner": "planner", "reflect": "reflect", "__end__": END}
)
graph.add_conditional_edges("reflect", route, path_map)   # reflect -> capability | planner | end
# planner and every capability already route via `route`; now `route` can also yield "reflect".
```

### New flow (matches the spec)

```
            ┌───────────────────────────── success ─────────────────────────────┐
            │                                                                    ▼
START → profile → planner ─[route]→ capability ─[route]→ (next capability) … → END
                                        │ fail (plan_error set)
                                        ▼
                                    reflect  ──── repair applied & changed ──┐
                                        │                                    │
                          no repair /   │  budget exhausted /                │
                          no change     │  repair failed                     │
                                        ▼                                    ▼
                                    planner  ◀───── escalate_to_planner   capability (retry)
                             (modify plan / swap model via
                              planner_hint / request input / abort)
```

The planner's existing replan branch is extended: when it is entered via
`escalate_to_planner`, it reads `reflection_history` and the stashed `planner_hint`, and may
(a) swap the model / shrink the search space (honouring `planner_hint.requires_model_swap` /
`reduce_search_space`), (b) modify or reorder the plan / choose another capability, or (c)
fall through to the current `needs_input` behaviour. Model-swap is a **planner** action driven
by the hint — never a repair. Its bounded `replan_attempts` budget is untouched — so the
system now has **three independent, bounded budgets**: `reflection_attempts` (per capability),
`repair_attempts` (per run), and `replan_attempts` (planner).

---

## 5. Step-by-step implementation plan

Ordered so every step lands with tests and the pipeline stays green throughout. Each step is
independently shippable.

### Phase A — models & taxonomy (no wiring, pure & safe)
1. `reflection/models.py` — the Pydantic models above (`FailureCategory`, `PlannerHint`,
   `Diagnosis`, `RepairSpec`, `RepairResult`, `ReflectionRecord`). Unit test: round-trip
   `.model_dump()` / re-parse; enum coverage; `planner_hint` defaults to `None`.
2. `reflection/taxonomy.py` — `classify()` with rules for the known failures. Unit test with
   raw strings for each category, **including the XGBoost `ValueError: could not convert
   string to float: 'Yes'` → `TARGET_ENCODING_ERROR`** golden case, and an **unknown-message
   test** asserting the result is `UNKNOWN_ERROR` with confidence `< min_confidence` (so the
   agent routes to the LLM rather than treating it as a final verdict).

### Phase B — repair primitives (pure dataframe work)
3. `tools/repair_ops.py` — each primitive (`label_encode_target`, `coerce_numeric`,
   `impute_missing_values`, `drop_constant_columns`, `remove_invalid_rows`,
   `convert_datetime_columns`, `validate_schema`). Signature mirrors `tools/cleaning.py`:
   `(in_path, out_path, …) -> dict`. Unit-test each on a tiny fixture parquet — this is where
   correctness is proven, in isolation from LangGraph.
4. `reflection/repairs.py` + `repair_registry.py` — wrap each primitive as a
   `RepairCapability` (parse `Diagnosis.evidence` → call the primitive → build `RepairResult`).
   Register them. Unit test: given a `Diagnosis`, the runner rewrites the fixture artifact and
   returns `changed=True`.

### Phase C — diagnoser & selection
5. `reflection/diagnoser.py` — heuristic-first `ReflectionAgent`, LLM fallback behind
   `min_confidence`, under `diagnosis_timeout_s`. Test the heuristic path with `FakeLLMClient`
   asserting **zero** LLM calls on a high-confidence match; test the fallback path (low-confidence
   / unknown message) asserting one call and a schema-valid `Diagnosis`; test that a
   timeout/`LLMError` degrades to the heuristic Diagnosis without raising.
6. `select_repair()` → `RepairPlan`. Test: suggested-repair wins; else category candidate;
   else planning concern (`MODEL_CONFIGURATION_ERROR` / `RESOURCE_LIMIT` or `planner_hint` set)
   → `escalate=True` with the hint; else escalate with no hint.

### Phase D — the node & config
7. `core/config.py` — add `reflection_attempts: int = 2`, `repair_attempts: int = 4`,
   `diagnosis_timeout_s: float = 20.0`.
8. `reflection/node.py` — `reflect_node`, `_escalate`, `_record`. Unit test the node directly
   (as `test_planner_retry.py` tests `planner_node`) with a `FakeLLMClient` and a fixture
   state: (a) successful repair → `pending_retry_capability` set, error cleared;
   (b) budget exhausted → `escalate_to_planner`; (c) no-change repair → escalate;
   (d) planning-concern diagnosis → `escalate_to_planner` with `planner_hint` stashed.

### Phase E — graph wiring (the migration)
9. `graph/state.py` — add the reflection control keys.
10. `graph/builder.py` — register `reflect`, add to `path_map`, add its conditional edge; add
    `failed_exc_type` + `pending_retry_capability: None` to the wrapper.
11. `planner/planner_node.py` — `route()` now yields `"reflect"` on `plan_error` and honours
    `pending_retry_capability` / `escalate_to_planner`; `planner_node` reads
    `reflection_history` on the escalation branch.

**Migration note (behaviour-preserving default):** if `REPAIR_REGISTRY` is empty or a
diagnosis maps to no repair, `reflect_node` immediately escalates — i.e. the system degrades
to *exactly today's behaviour* (`plan_error` → planner → bounded replan → `needs_input`). So
Phase E can merge before every repair exists; repairs then light up incrementally as Phase B/C
land. There is no flag-day cutover.

### Phase F — integration & end-to-end
12. `test_reflection_pipeline.py` — a full graph run over a fixture dataset whose target is
    `Yes/No`. Assert: training fails once → `reflect` runs → `label_encode_target` applied →
    training re-runs → run completes `completed`, and `reflection_history` has exactly one
    record with `retry_succeeded=True`.
13. Regression: existing `test_router_resume.py` / `test_planner_retry.py` still pass
    unchanged (the failure→planner path still exists, now via reflect's escalation).

### Testing strategy summary
- **Pure units** (taxonomy, repair_ops, models) — the bulk of correctness, no LangGraph, no LLM.
- **Node units** — `reflect_node` and `route` tested as pure `state -> dict` functions, the
  pattern already established in `test_planner_retry.py` and `test_router_resume.py`.
- **Fake LLM** — `FakeLLMClient` for the diagnoser's fallback path; assert call counts to prove
  heuristics avoid LLM cost on known failures.
- **One integration test** per repair-worthy failure mode, driving the real compiled graph.
- **Budget/termination tests** — assert the three budgets each bound the loop and that an
  unfixable failure still terminates in bounded steps (guards against an infinite
  repair↔retry cycle; the `recursion_limit=30` backstop in `runs.py` remains as a final net).

---

## Design guarantees checklist

- **Clean, layered separation** — models depend on nothing; repair primitives are pure
  dataframe tools; only `node.py` touches `RunState`. ✔
- **Pydantic for all structured data** — `Diagnosis`, `PlannerHint`, `RepairSpec`,
  `RepairResult`, `ReflectionRecord`; stored via `.model_dump()` like every existing report. ✔
- **LangGraph nodes** — one new `reflect` node; repair execution runs inside it via the
  registry. ✔
- **Registry-based, extensible repairs; no `if/else` in the executor** — `REPAIR_REGISTRY` +
  `candidates_for`; adding a repair is one registration. ✔
- **Repairs vs planning cleanly separated** — `REPAIR_REGISTRY` holds only artifact-level
  deterministic repairs (fix an artifact → retry same capability → return `RepairResult`).
  Strategy changes (model swap, search-space reduction, capability choice, abort) are carried
  by `Diagnosis.planner_hint` and owned by the planner — never modelled as repairs. ✔
- **Robust diagnosis** — unknown exceptions fall back to the LLM (low-confidence
  `UNKNOWN_ERROR` is a routing signal, not a terminal verdict); the LLM diagnosis runs under
  `diagnosis_timeout_s` and degrades to the heuristic verdict, so a hung provider can't stall
  the graph. ✔
- **Never edits/generates source** — repairs only rewrite on-disk *data artifacts* and flip
  documented spec/state knobs the tools already read. ✔
- **Bounded** — three independent budgets (`reflection_attempts`, `repair_attempts`,
  `replan_attempts`) + the graph `recursion_limit`. ✔
- **Persisted repair history** — `reflection_history: list[ReflectionRecord]` checkpointed to
  SQLite after every node and surfaced to the planner on escalation. ✔
```
