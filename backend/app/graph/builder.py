"""Pipeline graph assembly.

Current shape (linear, nine nodes):

    START -> profile -> insights -> problem_spec -> cleaning
          -> feature_plan -> features -> training -> evaluation
          -> summarize -> END

State is checkpointed to SQLite after every node, so an interrupted run
resumes without re-executing completed steps. The planner/executor loop
planned for the next phase will replace the fixed ordering; the node
contract stays the same.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.graph.nodes import (
    cleaning_node,
    evaluation_node,
    feature_plan_node,
    features_node,
    insights_node,
    problem_spec_node,
    profile_node,
    summarize_node,
    training_node,
)
from app.graph.state import RunState


@lru_cache(maxsize=1)
def build_graph():
    """Compile the pipeline graph (one instance per process)."""
    graph = StateGraph(RunState)

    graph.add_node("profile", profile_node)
    graph.add_node("insights", insights_node)
    graph.add_node("problem_spec", problem_spec_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("feature_plan", feature_plan_node)
    graph.add_node("features", features_node)
    graph.add_node("training", training_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "insights")
    graph.add_edge("insights", "problem_spec")
    graph.add_edge("problem_spec", "cleaning")
    graph.add_edge("cleaning", "feature_plan")
    graph.add_edge("feature_plan", "features")
    graph.add_edge("features", "training")
    graph.add_edge("training", "evaluation")
    graph.add_edge("evaluation", "summarize")
    graph.add_edge("summarize", END)

    # check_same_thread=False: FastAPI serves requests from multiple threads.
    conn = sqlite3.connect(
        settings.data_dir / "checkpoints.db", check_same_thread=False
    )
    return graph.compile(checkpointer=SqliteSaver(conn))
