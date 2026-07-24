"""Final report assembler.

Ties every stage of a run into one consolidated report: a structured dict for
the API/frontend, and a rendered Markdown file for download. Pure string
templating — no reporting libraries, no plotting. The structured pieces already
exist in the run state; this tool just arranges and renders them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FinalReport(BaseModel):
    run_id: str
    dataset_id: str
    problem: dict[str, Any]
    dataset_overview: dict[str, Any]
    cleaning: dict[str, Any]
    features: dict[str, Any]
    model: dict[str, Any]
    evaluation: dict[str, Any]
    explainability: dict[str, Any]
    recommendations: dict[str, Any]
    narrative: str | None = Field(default=None, description="Plain-language summary")
    report_path: str


def assemble_report(state_slice: dict[str, Any], artifacts_dir: Path) -> dict[str, Any]:
    """Build the structured report and write report.md; return the report dict."""
    profile = state_slice["profile"]
    spec = state_slice["problem_spec"]
    cleaning = state_slice["cleaning_report"]
    features = state_slice["feature_report"]
    training = state_slice["training_report"]
    evaluation = state_slice["evaluation_report"]
    explanation = state_slice["explanation_report"]
    recommendations = state_slice["recommendations"]
    insights = state_slice.get("insights")
    summary = state_slice.get("summary")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "report.md"

    report = FinalReport(
        run_id=state_slice["run_id"],
        dataset_id=state_slice["dataset_id"],
        problem={
            "problem_type": spec["problem_type"],
            "target_column": spec["target_column"],
        },
        dataset_overview={"n_rows": profile["n_rows"], "n_cols": profile["n_cols"]},
        cleaning=cleaning,
        features=features,
        model={
            "best_model": training["best_model"],
            "metric": training["metric"],
            "leaderboard": training["results"],
        },
        evaluation=evaluation,
        explainability={
            "method": explanation["method"],
            "top_features": explanation.get("top_features", []),
        },
        recommendations=recommendations,
        narrative=summary,
        report_path=str(report_path),
    )

    report_path.write_text(_render_markdown(report, insights), encoding="utf-8")
    return report.model_dump()


def _render_markdown(r: FinalReport, insights: dict[str, Any] | None) -> str:
    lines: list[str] = [
        f"# Data Science Report — run {r.run_id}",
        "",
        f"**Dataset:** `{r.dataset_id}` — {r.dataset_overview['n_rows']} rows × "
        f"{r.dataset_overview['n_cols']} columns",
        f"**Problem:** {r.problem['problem_type']} — predicting "
        f"`{r.problem['target_column']}`",
    ]

    if r.narrative:
        lines += ["", "## Summary", "", r.narrative]

    if insights and insights.get("insights"):
        lines += ["", "## Data quality findings", ""]
        for item in insights["insights"]:
            lines.append(f"- **{item['title']}** ({item['severity']}): {item['detail']}")

    lines += [
        "",
        "## Data preparation",
        "",
        f"- Cleaning: removed {r.cleaning['rows_removed']} rows; dropped columns "
        f"{r.cleaning['dropped_columns'] or 'none'}.",
        f"- Features: {r.features['n_features_before']} → "
        f"{r.features['n_features_after']} columns "
        f"(expanded {r.features['expanded_datetime_columns'] or 'none'}, "
        f"log-transformed {r.features['log_transformed_columns'] or 'none'}).",
        "",
        "## Model leaderboard (cross-validated)",
        "",
        f"Metric: `{r.model['metric']}` — best model: **{r.model['best_model']}**",
        "",
    ]
    for row in r.model["leaderboard"]:
        lines.append(f"- {row['model']}: {row['cv_mean']} (± {row['cv_std']})")

    lines += ["", "## Held-out evaluation", ""]
    for name, value in r.evaluation.get("metrics", {}).items():
        lines.append(f"- {name}: {value}")

    lines += ["", f"## Top drivers ({r.explainability['method']})", ""]
    top = r.explainability.get("top_features", [])
    if top:
        for f in top:
            arrow = f" ({f['direction']})" if f.get("direction") else ""
            lines.append(f"- {f['feature']}: {f['importance']}{arrow}")
    else:
        lines.append("- Feature importances were unavailable for this run.")

    lines += ["", "## Recommendations", ""]
    if r.recommendations.get("narrative"):
        lines += [r.recommendations["narrative"], ""]
    for rec in r.recommendations.get("recommendations", []):
        lines.append(
            f"- **{rec['title']}** (confidence: {rec['confidence']}) — "
            f"{rec['detail']} _Expected impact: {rec['expected_impact']}_"
        )

    lines.append("")
    return "\n".join(lines)
