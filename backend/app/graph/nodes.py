"""Pipeline nodes.

Each node is a small function: it reads what it needs from the state,
delegates the real work to a tool or agent, and returns only the keys it
produces. LangGraph merges partial updates into the shared state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.agents.eda_agent import EDAInsightsAgent
from app.agents.feature_agent import FeatureEngineeringAgent
from app.agents.problem_agent import ProblemSpecAgent
from app.agents.recommendations_agent import RecommendationsAgent
from app.core.config import settings
from app.graph.state import RunState
from app.llm.factory import get_llm_client
from app.tools.cleaning import clean_dataset
from app.tools.evaluation import evaluate_model
from app.tools.explain import explain_model
from app.tools.features import engineer_features
from app.tools.profiler import profile_dataset
from app.tools.report import assemble_report
from app.tools.training import CV_FOLDS, train_models

logger = logging.getLogger(__name__)


def _dataset_dir(state: RunState) -> Path:
    return settings.datasets_dir / state["dataset_id"]


def _artifacts_dir(state: RunState) -> Path:
    return settings.artifacts_dir / state["run_id"]


def profile_node(state: RunState) -> dict:
    """Profile the dataset (deterministic, no LLM)."""
    ds_dir = _dataset_dir(state)

    profile_path = ds_dir / "profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
    else:
        raw_files = list(ds_dir.glob("raw.*"))
        if not raw_files:
            raise FileNotFoundError(f"No raw dataset file for {state['dataset_id']}")
        profile = profile_dataset(raw_files[0])
        profile_path.write_text(json.dumps(profile, indent=2))

    logger.info(
        "run=%s profiled: %d rows x %d cols",
        state["run_id"], profile["n_rows"], profile["n_cols"],
    )
    return {"profile": profile}


def insights_node(state: RunState) -> dict:
    """Generate insights from the profile (LLM call)."""
    agent = EDAInsightsAgent(llm=get_llm_client())
    report = agent.run(state["profile"])
    logger.info("run=%s insights: %d findings", state["run_id"], len(report.insights))
    return {"insights": report.model_dump()}


def problem_spec_node(state: RunState) -> dict:
    """Decide problem type, target, and columns to drop (LLM call)."""
    agent = ProblemSpecAgent(llm=get_llm_client())
    spec = agent.run(state["problem_text"], state["profile"])
    logger.info(
        "run=%s spec: %s on '%s', dropping %s",
        state["run_id"], spec.problem_type, spec.target_column, spec.drop_columns,
    )
    return {"problem_spec": spec.model_dump()}


def cleaning_node(state: RunState) -> dict:
    """Apply the cleaning plan (deterministic tool)."""
    spec = state["problem_spec"]
    raw_files = list(_dataset_dir(state).glob("raw.*"))

    report = clean_dataset(
        raw_path=raw_files[0],
        out_path=_artifacts_dir(state) / "cleaned.parquet",
        target_column=spec["target_column"],
        drop_columns=spec["drop_columns"],
        drop_duplicates=spec["drop_duplicates"],
    )
    logger.info(
        "run=%s cleaned: %d -> %d rows",
        state["run_id"], report["rows_before"], report["rows_after"],
    )
    return {"cleaning_report": report}


def feature_plan_node(state: RunState) -> dict:
    """Decide which feature transforms to apply (LLM call).

    The plan is based on a profile of the *cleaned* data, not the raw
    profile, so skew and dtypes reflect the columns that actually survive
    into modeling.
    """
    cleaned_path = Path(state["cleaning_report"]["cleaned_path"])
    cleaned_profile = profile_dataset(cleaned_path)

    agent = FeatureEngineeringAgent(llm=get_llm_client())
    plan = agent.run(state["problem_spec"], cleaned_profile)
    logger.info(
        "run=%s feature plan: expand=%s, log=%s, drop=%s",
        state["run_id"], plan.datetime_columns,
        plan.log_transform_columns, plan.drop_columns,
    )
    return {"feature_plan": plan.model_dump()}


def features_node(state: RunState) -> dict:
    """Apply the feature plan (deterministic tool)."""
    spec = state["problem_spec"]
    plan = state["feature_plan"]
    report = engineer_features(
        cleaned_path=Path(state["cleaning_report"]["cleaned_path"]),
        out_path=_artifacts_dir(state) / "features.parquet",
        target_column=spec["target_column"],
        datetime_columns=plan["datetime_columns"],
        log_transform_columns=plan["log_transform_columns"],
        drop_columns=plan["drop_columns"],
    )
    logger.info(
        "run=%s features: %d -> %d columns",
        state["run_id"], report["n_features_before"], report["n_features_after"],
    )
    return {"feature_report": report}


def training_node(state: RunState) -> dict:
    """Train candidate models with cross-validation (deterministic tool)."""
    spec = state["problem_spec"]
    report = train_models(
        data_path=Path(state["feature_report"]["features_path"]),
        target_column=spec["target_column"],
        problem_type=spec["problem_type"],
        artifacts_dir=_artifacts_dir(state),
        cv_folds=spec.get("cv_folds", CV_FOLDS),
    )
    logger.info(
        "run=%s best model: %s (%s=%.4f)",
        state["run_id"], report["best_model"], report["metric"], report["best_score"],
    )
    return {"training_report": report}


def evaluation_node(state: RunState) -> dict:
    """Score the winning model on a held-out test split (deterministic tool)."""
    spec = state["problem_spec"]
    report = evaluate_model(
        data_path=Path(state["feature_report"]["features_path"]),
        model_path=Path(state["training_report"]["model_path"]),
        target_column=spec["target_column"],
        problem_type=spec["problem_type"],
    )
    logger.info("run=%s eval: %s", state["run_id"], report["metrics"])
    return {"evaluation_report": report}


def explain_node(state: RunState) -> dict:
    """Compute global feature importances for the trained model (tool)."""
    spec = state["problem_spec"]
    training = state["training_report"]
    report = explain_model(
        data_path=Path(state["feature_report"]["features_path"]),
        model_path=Path(training["model_path"]),
        target_column=spec["target_column"],
        problem_type=spec["problem_type"],
        best_model=training["best_model"],
    )
    logger.info(
        "run=%s explain: %s, %d features",
        state["run_id"], report["method"], len(report["top_features"]),
    )
    return {"explanation_report": report}


def recommendations_node(state: RunState) -> dict:
    """Turn metrics + drivers into business recommendations (LLM call)."""
    agent = RecommendationsAgent(llm=get_llm_client())
    report = agent.run(
        state["problem_spec"],
        state["evaluation_report"],
        state["explanation_report"],
    )
    logger.info(
        "run=%s recommendations: %d items", state["run_id"], len(report.recommendations)
    )
    return {"recommendations": report.model_dump()}


def report_node(state: RunState) -> dict:
    """Assemble the consolidated final report and render report.md (tool)."""
    report = assemble_report(
        state_slice={
            "run_id": state["run_id"],
            "dataset_id": state["dataset_id"],
            "profile": state["profile"],
            "insights": state.get("insights"),
            "problem_spec": state["problem_spec"],
            "cleaning_report": state["cleaning_report"],
            "feature_report": state["feature_report"],
            "training_report": state["training_report"],
            "evaluation_report": state["evaluation_report"],
            "explanation_report": state["explanation_report"],
            "recommendations": state["recommendations"],
            "summary": state.get("summary"),
        },
        artifacts_dir=_artifacts_dir(state),
    )
    logger.info("run=%s report written: %s", state["run_id"], report["report_path"])
    return {"report": report}


def summarize_node(state: RunState) -> dict:
    """Assemble the final run summary (plain Python, no LLM)."""
    profile = state["profile"]
    insights = state.get("insights")  # optional capability — may be skipped
    spec = state["problem_spec"]
    cleaning = state["cleaning_report"]
    features = state["feature_report"]
    training = state["training_report"]
    evaluation = state["evaluation_report"]

    lines = [f"Dataset: {profile['n_rows']} rows x {profile['n_cols']} columns."]
    if insights:
        lines.append(insights["summary"])
    lines += [
        "",
        f"Problem: {spec['problem_type']} — predicting '{spec['target_column']}'.",
        f"Cleaning: removed {cleaning['rows_removed']} rows, "
        f"dropped columns {cleaning['dropped_columns'] or 'none'}.",
        f"Features: {features['n_features_before']} -> {features['n_features_after']} columns "
        f"(expanded {features['expanded_datetime_columns'] or 'none'}, "
        f"log-transformed {features['log_transformed_columns'] or 'none'}).",
        "",
        "Model leaderboard (cross-validated):",
    ]
    for row in training["results"]:
        lines.append(f"  {row['model']}: {row['cv_mean']} (+/- {row['cv_std']})")
    lines.append("")
    lines.append(f"Best model: {training['best_model']}")
    lines.append(f"Held-out test metrics: {evaluation['metrics']}")

    return {"summary": "\n".join(lines), "status": "completed"}
