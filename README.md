
# Project Title

A brief description of what this project does and who it's for

# AI Data Science Platform

> Working title — final project name TBD.

An autonomous AI data scientist: upload a dataset, describe a business
problem, and a multi-agent workflow (LangGraph) performs the full data
science lifecycle — EDA, cleaning, feature engineering, model training,
evaluation, explainability, business recommendations, and a final report
with a downloadable model.

## Structure

```
├── backend/     # FastAPI + LangGraph + ML tool layer
└── frontend/    # UI (stack TBD — talks to the backend via REST)
```

## Backend — run locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

Run the tests:

```bash
cd backend
pytest
```

Configuration goes in `backend/.env`:

```
APP_OPENROUTER_API_KEY=sk-or-...
APP_LLM_PROVIDER=openrouter
APP_LLM_MODEL=anthropic/claude-sonnet-4.5
```

## Architecture

- **Static graph, dynamic plan** — LangGraph nodes are fixed; a Planner
  Agent generates a per-dataset ExecutionPlan (JSON), validated by rules.
- **Agents use tools, never generate code** — all ML work lives in a
  deterministic, testable tool layer (pandas/sklearn/XGBoost/Optuna/SHAP).
- **Registry-bound planning** — the planner can only pick registered
  capabilities; a DAG validator checks requires/produces.
- **State holds references** — dataframes and models stay on disk; the
  graph state carries paths and summaries only.
- **Checkpoint after every node** — runs are resumable; retries are
  bounded (plan: 3, replan: 2) with fallback to human input.

## Build phases

1. ✅ Backend foundation: upload + dataset profiler
2. ✅ LLM abstraction (provider-independent) + EDA agent + 3-node LangGraph pipeline
3. ✅ ML tools + agents: cleaning, features, training, evaluation
4. ✅ Dynamic Planner + capability registry + validator + executor
5. 🚧 SHAP + recommendations + report (backend ✅) + frontend (pending)
