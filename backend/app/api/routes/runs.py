"""Run management API.

A run represents one pipeline execution: one dataset, one problem
statement, one pass through the graph. Run records are persisted as JSON
under the storage directory.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import settings
from app.graph.builder import build_graph

router = APIRouter()


class CreateRunRequest(BaseModel):
    dataset_id: str
    problem_text: str = ""


class RunResponse(BaseModel):
    run_id: str
    dataset_id: str
    status: str
    problem_text: str
    created_at: str
    summary: str | None = None
    execution_plan: list[str] | None = None
    plan_reasoning: str | None = None
    warnings: list[str] | None = None
    insights: dict[str, Any] | None = None
    problem_spec: dict[str, Any] | None = None
    cleaning_report: dict[str, Any] | None = None
    feature_report: dict[str, Any] | None = None
    training_report: dict[str, Any] | None = None
    evaluation_report: dict[str, Any] | None = None
    explanation_report: dict[str, Any] | None = None
    recommendations: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    error: str | None = None


def _runs_dir(run_id: str) -> Path:
    d = settings.data_dir / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("", response_model=RunResponse)
def create_run(req: CreateRunRequest) -> RunResponse:
    ds_dir = settings.datasets_dir / req.dataset_id
    if not (ds_dir / "metadata.json").exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    run_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()

    initial_state = {
        "run_id": run_id,
        "dataset_id": req.dataset_id,
        "problem_text": req.problem_text,
        "status": "running",
    }

    graph = build_graph()
    # recursion_limit backstops the router loop: it caps total node visits so a
    # bug that fails to advance the plan cursor trips the limit instead of
    # looping forever.
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 30}

    try:
        final_state = graph.invoke(initial_state, config=config)
    except Exception as exc:
        # A node failure produces a failed run record, not an HTTP 500 —
        # the run itself is the unit the frontend tracks.
        record = {
            **initial_state,
            "status": "failed",
            "error": str(exc),
            "created_at": created_at,
        }
        (_runs_dir(run_id) / "run.json").write_text(json.dumps(record, indent=2))
        return RunResponse(
            run_id=run_id,
            dataset_id=req.dataset_id,
            status="failed",
            problem_text=req.problem_text,
            created_at=created_at,
            error=str(exc),
        )

    record = {
        "run_id": run_id,
        "dataset_id": req.dataset_id,
        "problem_text": req.problem_text,
        "status": final_state.get("status", "completed"),
        "created_at": created_at,
        "summary": final_state.get("summary"),
        "execution_plan": final_state.get("execution_plan"),
        "plan_reasoning": final_state.get("plan_reasoning"),
        "warnings": final_state.get("warnings"),
        "insights": final_state.get("insights"),
        "problem_spec": final_state.get("problem_spec"),
        "cleaning_report": final_state.get("cleaning_report"),
        "feature_report": final_state.get("feature_report"),
        "training_report": final_state.get("training_report"),
        "evaluation_report": final_state.get("evaluation_report"),
        "explanation_report": final_state.get("explanation_report"),
        "recommendations": final_state.get("recommendations"),
        "report": final_state.get("report"),
    }
    (_runs_dir(run_id) / "run.json").write_text(json.dumps(record, indent=2))

    return RunResponse(**record)


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    run_path = settings.data_dir / "runs" / run_id / "run.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse(**json.loads(run_path.read_text()))


@router.get("/{run_id}/model")
def download_model(run_id: str) -> FileResponse:
    """Download the trained model produced by a run."""
    model_path = settings.artifacts_dir / run_id / "model.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model for this run")
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"model_{run_id}.joblib",
    )


@router.get("/{run_id}/report")
def download_report(run_id: str) -> FileResponse:
    """Download the consolidated Markdown report produced by a run."""
    report_path = settings.artifacts_dir / run_id / "report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No report for this run")
    return FileResponse(
        report_path,
        media_type="text/markdown",
        filename=f"report_{run_id}.md",
    )


@router.get("", response_model=list[RunResponse])
def list_runs() -> list[RunResponse]:
    runs_root = settings.data_dir / "runs"
    if not runs_root.exists():
        return []
    out = []
    for d in sorted(runs_root.iterdir(), reverse=True):
        p = d / "run.json"
        if p.exists():
            out.append(RunResponse(**json.loads(p.read_text())))
    return out
