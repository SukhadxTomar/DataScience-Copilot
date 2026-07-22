"""Insights API — run the EDA insights agent on an uploaded dataset."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from app.agents.eda_agent import EDAInsightsAgent, InsightReport
from app.core.config import settings
from app.llm.base import LLMAuthError, LLMError
from app.llm.factory import get_llm_client

router = APIRouter()


@router.post("/{dataset_id}/insights", response_model=InsightReport)
def generate_insights(dataset_id: str) -> Any:
    ds_dir = settings.datasets_dir / dataset_id
    profile_path = ds_dir / "profile.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Serve cached insights if they were generated before.
    insights_path = ds_dir / "insights.json"
    if insights_path.exists():
        return InsightReport.model_validate_json(insights_path.read_text())

    profile = json.loads(profile_path.read_text())
    agent = EDAInsightsAgent(llm=get_llm_client())

    try:
        report = agent.run(profile)
    except LLMAuthError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    insights_path.write_text(report.model_dump_json(indent=2))
    return report
