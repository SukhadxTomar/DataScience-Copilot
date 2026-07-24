"""Business recommendations agent.

Reads the modeling results — the problem, the held-out metrics, and the top
feature importances — and turns them into concrete, business-actionable
recommendations. Like the EDA agent, it reasons only over tool output
(numbers and feature names), never raw data, and grounds every claim in the
figures it was given.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm.base import LLMClient

MAX_RECOMMENDATIONS = 6


class Recommendation(BaseModel):
    title: str = Field(description="Short imperative headline, e.g. 'Target high-risk plans'")
    detail: str = Field(description="1-3 sentences explaining the action in business terms")
    expected_impact: str = Field(
        description="Qualitative expected outcome, grounded in the given metrics/features"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How strongly the results support this recommendation"
    )


class RecommendationsReport(BaseModel):
    narrative: str = Field(description="2-4 sentence overall assessment of the model and what it means")
    recommendations: list[Recommendation] = Field(
        description="3-6 actionable items, highest impact first"
    )


_SYSTEM_PROMPT = """You are a senior data science consultant briefing a business \
stakeholder on what to do with a freshly trained model.

You will receive: the problem (type + target), the held-out evaluation metrics, \
and the model's top feature importances (the drivers of its predictions). \
Produce a short narrative and concrete recommendations.

Rules:
- Ground EVERY statement in the provided metrics and feature list. Never invent \
numbers or name features that are not in the list.
- Turn drivers into action: if a feature strongly influences the outcome, \
recommend what the business should do about it.
- Be honest about reliability: if the metrics are weak, say so and set \
confidence accordingly.
- Write for a decision-maker, not a statistician. No code, no jargon dumps."""


class RecommendationsAgent:
    """Produces a RecommendationsReport from the modeling results."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(
        self,
        problem_spec: dict[str, Any],
        evaluation_report: dict[str, Any],
        explanation_report: dict[str, Any],
    ) -> RecommendationsReport:
        user_prompt = (
            f"Problem: {problem_spec['problem_type']} — predicting "
            f"'{problem_spec['target_column']}'.\n\n"
            f"Held-out metrics:\n{json.dumps(evaluation_report.get('metrics', {}), indent=2)}\n\n"
            f"Top feature importances ({explanation_report.get('method', 'n/a')}):\n"
            f"{json.dumps(explanation_report.get('top_features', []), indent=2)}"
        )
        report = self._llm.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            schema=RecommendationsReport,
        )
        # Keep the briefing tight — cap the number of items.
        report.recommendations = report.recommendations[:MAX_RECOMMENDATIONS]
        return report
