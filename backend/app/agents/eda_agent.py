"""EDA insights agent.

Turns a dataset profile into business-friendly findings. The agent reads
tool output (the compact profile), never raw data, and its response is
schema-enforced through the LLM client.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm.base import LLMClient


class Insight(BaseModel):
    title: str = Field(description="Short headline, e.g. 'High missing values in income'")
    detail: str = Field(description="1-3 sentence explanation in plain business language")
    severity: Literal["info", "warning", "critical"] = Field(
        description="critical = blocks modeling, warning = needs handling, info = good to know"
    )
    columns: list[str] = Field(default_factory=list, description="Columns involved")


class InsightReport(BaseModel):
    summary: str = Field(description="2-3 sentence overview of the dataset")
    insights: list[Insight] = Field(description="5-10 concrete findings, most important first")
    suggested_target: str | None = Field(
        default=None,
        description="Column that looks like the prediction target, if any",
    )


_SYSTEM_PROMPT = """You are a senior data scientist reviewing a dataset profile \
for a business stakeholder.

You will receive a JSON profile of a dataset (column types, missing values, \
statistics, warnings). Produce concrete, actionable insights.

Rules:
- Base every insight ONLY on the profile data. Never invent numbers.
- Prefer specific over generic: "monthly_charges has 8.1% missing values" \
beats "some data is missing".
- Flag data quality issues that would break or bias a machine learning model.
- If a column looks like a prediction target (binary/categorical outcome, \
names like churned/fraud/converted), point it out.
- Write for a smart business person, not a statistician."""


class EDAInsightsAgent:
    """Produces an InsightReport from a dataset profile."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, profile: dict[str, Any]) -> InsightReport:
        user_prompt = (
            "Analyze this dataset profile and return your findings:\n\n"
            f"{json.dumps(profile, indent=2)}"
        )
        return self._llm.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            schema=InsightReport,
        )
