"""Problem specification agent.

Combines the user's problem statement with the dataset profile to decide
what kind of ML problem this is, which column is the target, and which
columns should be excluded before modeling. The decision is data for the
pipeline — deterministic tools execute it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm.base import LLMClient


class ProblemSpec(BaseModel):
    problem_type: Literal["classification", "regression"] = Field(
        description="classification for categorical/binary outcomes, regression for continuous ones"
    )
    target_column: str = Field(description="Exact name of the column to predict")
    drop_columns: list[str] = Field(
        default_factory=list,
        description="Columns to exclude before modeling: identifiers, constants, leakage risks",
    )
    drop_duplicates: bool = Field(description="Whether duplicate rows should be removed")
    reasoning: str = Field(description="2-3 sentences explaining these choices")


_SYSTEM_PROMPT = """You are a senior data scientist setting up a machine \
learning problem.

You will receive the user's business goal and a JSON profile of their \
dataset. Decide:
1. problem_type — classification (categorical/binary target) or regression \
(continuous target).
2. target_column — must be an EXACT column name from the profile. Prefer \
the user's stated goal; fall back to the most plausible outcome column.
3. drop_columns — identifiers (role_hint id_like), constant columns, and \
any column that would leak the answer into the features.
4. drop_duplicates — true if the profile reports duplicate rows.

Never invent column names. Never put the target in drop_columns."""


class ProblemSpecAgent:
    """Produces a ProblemSpec from the problem statement and profile."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, problem_text: str, profile: dict[str, Any]) -> ProblemSpec:
        user_prompt = (
            f"Business goal: {problem_text or 'not specified — infer from the data'}\n\n"
            f"Dataset profile:\n{json.dumps(profile, indent=2)}"
        )
        spec = self._llm.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            schema=ProblemSpec,
        )

        # Guard against hallucinated column names before anything downstream runs.
        valid_columns = {c["name"] for c in profile["columns"]}
        if spec.target_column not in valid_columns:
            raise ValueError(
                f"LLM chose target '{spec.target_column}' which is not in the dataset"
            )
        spec.drop_columns = [
            c for c in spec.drop_columns
            if c in valid_columns and c != spec.target_column
        ]
        return spec
