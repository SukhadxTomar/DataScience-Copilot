"""Planner agent.

Given the dataset profile and the user's goal, chooses which registered
capabilities to run and in what order — an ExecutionPlan. The agent is
registry-bound: the prompt lists exactly the capabilities it may pick from,
with their requires/produces, and the agent sanitizes the response down to
known names. Structural soundness (are the requires satisfiable in this
order?) is proven separately by the validator, whose errors can be fed back
here via ``feedback`` on a retry.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.llm.base import LLMClient
from app.planner.registry import REGISTRY, describe_for_prompt


class ExecutionPlan(BaseModel):
    steps: list[str] = Field(
        description="Ordered capability names to run, each an EXACT name from "
        "the provided capability list"
    )
    reasoning: str = Field(description="2-3 sentences justifying the plan")


_SYSTEM_PROMPT = """You are the planner for an automated machine-learning \
pipeline. You decide which capabilities run and in what order for THIS dataset.

You may ONLY choose from the capabilities listed in the user message, using \
their EXACT names. Output an ordered list in `steps`.

Rules:
- Order the steps so that, for each one, every artifact it `requires` has \
already been `produced` by an earlier step. The bootstrap artifacts \
(run_id, dataset_id, problem_text, profile) already exist before step 1.
- The plan MUST end having produced `summary` and `evaluation_report` — i.e. \
it must run the full modeling chain through `summarize`.
- Do not repeat a capability.
- `insights` is optional (nothing else depends on it); include it when the \
dataset has notable data-quality issues worth reporting, otherwise you may \
skip it to save an LLM call.
- Every other capability is required because later steps depend on its output."""


class PlannerAgent:
    """Produces an ExecutionPlan from the profile and problem statement."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(
        self,
        profile: dict[str, Any],
        problem_text: str,
        feedback: str | None = None,
    ) -> ExecutionPlan:
        user_parts = [
            "Available capabilities:",
            describe_for_prompt(),
            "",
            f"Business goal: {problem_text or 'not specified — infer from the data'}",
            "",
            "Dataset profile:",
            json.dumps(profile, indent=2),
        ]
        if feedback:
            user_parts += [
                "",
                "Your previous plan was rejected by the validator. Fix these "
                "problems and return a corrected plan:",
                feedback,
            ]

        plan = self._llm.complete(
            system=_SYSTEM_PROMPT,
            user="\n".join(user_parts),
            schema=ExecutionPlan,
        )

        # Cheap sanitation only — drop hallucinated names and duplicates,
        # preserving order. The validator does the authoritative structural check.
        seen: set[str] = set()
        cleaned: list[str] = []
        for name in plan.steps:
            if name in REGISTRY and name not in seen:
                seen.add(name)
                cleaned.append(name)
        plan.steps = cleaned
        return plan
