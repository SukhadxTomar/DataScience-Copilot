"""Pipeline graph assembly — static graph, dynamic plan.

The graph shape is fixed: a bootstrap ``profile`` step, a ``planner``, and one
node per registered capability. What varies per run is the *plan* — an ordered
list of capabilities the planner produces and a ``route`` function walks:

    START -> profile -> planner -> [route] -> <capability> -> [route] -> ... -> END

Each capability loops back through ``route`` (a conditional edge), which reads
the checkpointed ``execution_plan`` + ``plan_cursor`` to dispatch the next step
or end. Because the plan and cursor live in the checkpointed state, an
interrupted run resumes exactly where it stopped — without re-invoking the
(non-deterministic) planner. State is checkpointed to SQLite after every node.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.graph.nodes import profile_node
from app.graph.state import RunState
from app.planner.planner_node import planner_node, route
from app.planner.registry import REGISTRY, Capability

logger = logging.getLogger(__name__)


def _make_capability_node(cap: Capability):
    """Wrap a capability's runner so it advances the plan cursor.

    On success the produced keys are merged with an incremented cursor, so the
    checkpoint after this node already points at the next step. On failure the
    error is recorded *without* advancing the cursor, so ``route`` sends control
    to the planner for a bounded replan that retries this same step.
    """

    def _node(state: RunState) -> dict:
        try:
            produced = cap.runner(state)
        except Exception as exc:  # noqa: BLE001 — surfaced via state, not raised
            logger.warning("run=%s capability '%s' failed: %s",
                           state.get("run_id"), cap.name, exc)
            return {"plan_error": str(exc), "failed_capability": cap.name}
        cursor = state.get("plan_cursor", 0)
        return {**produced, "plan_cursor": cursor + 1}

    return _node


@lru_cache(maxsize=1)
def build_graph():
    """Compile the pipeline graph (one instance per process)."""
    graph = StateGraph(RunState)

    graph.add_node("profile", profile_node)
    graph.add_node("planner", planner_node)
    for name, cap in REGISTRY.items():
        graph.add_node(name, _make_capability_node(cap))

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "planner")

    # planner and every capability hand control back to route(), which returns
    # the next capability name, "planner" (replan after a runtime failure), or
    # "__end__".
    path_map = {name: name for name in REGISTRY} | {"planner": "planner", "__end__": END}
    graph.add_conditional_edges("planner", route, path_map)
    for name in REGISTRY:
        graph.add_conditional_edges(name, route, path_map)

    # check_same_thread=False: FastAPI serves requests from multiple threads.
    conn = sqlite3.connect(
        settings.data_dir / "checkpoints.db", check_same_thread=False
    )
    return graph.compile(checkpointer=SqliteSaver(conn))
