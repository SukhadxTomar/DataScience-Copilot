"""Feature engineering agent.

Reads the dataset profile and the already-decided problem spec, then
proposes a feature plan: which datetime columns to expand into numeric
parts, which skewed numeric columns to log-transform, and which extra
columns to drop. The plan is plain data — the deterministic feature tool
executes it. The agent never sees or touches raw dataframes.

Only leak-safe, row-wise transforms are offered here on purpose (see
``app.tools.features``); scaling/imputation/encoding stay in the training
Pipeline where they can be fitted per-fold.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.llm.base import LLMClient


class FeaturePlan(BaseModel):
    datetime_columns: list[str] = Field(
        default_factory=list,
        description="Datetime (or datetime-string) columns to expand into "
        "year/month/day/dayofweek parts. The original column is dropped.",
    )
    log_transform_columns: list[str] = Field(
        default_factory=list,
        description="Right-skewed, non-negative numeric columns to log-transform "
        "(e.g. income, charges, counts). Never the target.",
    )
    drop_columns: list[str] = Field(
        default_factory=list,
        description="Extra columns to drop that survived cleaning but add no "
        "signal (e.g. high-cardinality free text). Never the target.",
    )
    reasoning: str = Field(description="2-3 sentences explaining these choices")


_SYSTEM_PROMPT = """You are a senior data scientist engineering features for a \
machine learning model.

You will receive the problem spec (problem type and target column) and a JSON \
profile of the CLEANED dataset. Propose a feature plan using ONLY these \
leak-safe, row-wise transforms:

1. datetime_columns — columns whose dtype is 'datetime' or 'datetime_string'. \
Expanding them exposes seasonality (month, day-of-week) to the model.
2. log_transform_columns — numeric columns that look right-skewed and are \
non-negative. Read skew from the profile stats: mean much greater than median \
suggests a right tail (income, charges, counts, amounts). Do NOT log-transform \
the target, booleans, or anything with negative values.
3. drop_columns — columns that survived cleaning but carry little signal, such \
as high-cardinality free text (role_hint 'high_cardinality').

Rules:
- Use EXACT column names from the profile. Never invent columns.
- Never include the target column in any list.
- Do NOT propose scaling, imputation, or one-hot encoding — those happen later \
inside the model pipeline. Leave normal numeric and low-cardinality categorical \
columns untouched.
- If nothing is worth transforming, return empty lists and say so."""


class FeatureEngineeringAgent:
    """Produces a FeaturePlan from the problem spec and cleaned-data profile."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, problem_spec: dict[str, Any], profile: dict[str, Any]) -> FeaturePlan:
        user_prompt = (
            f"Problem type: {problem_spec['problem_type']}\n"
            f"Target column: {problem_spec['target_column']}\n\n"
            f"Cleaned dataset profile:\n{json.dumps(profile, indent=2)}"
        )
        plan = self._llm.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            schema=FeaturePlan,
        )

        # Guard against hallucinated column names and target leakage before the
        # deterministic tool runs. Filtering here keeps the tool's contract simple.
        valid_columns = {c["name"] for c in profile["columns"]}
        target = problem_spec["target_column"]

        def _clean(cols: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for c in cols:
                if c in valid_columns and c != target and c not in seen:
                    seen.add(c)
                    out.append(c)
            return out

        plan.datetime_columns = _clean(plan.datetime_columns)
        plan.log_transform_columns = _clean(plan.log_transform_columns)
        plan.drop_columns = _clean(plan.drop_columns)
        return plan
