# DataScience Copilot — Interview Prep

> A study of what this project actually does, written from reading the code — not the folder names.
> Every claim below points to real code. Where something is unclear or unfinished, it says so.

---

# Section 1 — Understand This Project

## What it does (plain version)

- You upload a dataset (CSV/Parquet/Excel) and type a plain-English problem, like "predict churn".
- The backend runs a full ML pipeline on its own: it profiles the data, figures out the problem type and target, cleans the data, builds features, trains several models, picks the best, scores it on held-out data, explains it, and writes a report.
- If a step fails partway through, the system tries to diagnose and fix the problem itself, then retries — instead of just crashing.

That last part (the self-fix loop) is the most interesting piece, and the part most worth understanding deeply.

## The problem it solves and who has it

- **Who:** someone with a tabular dataset and a question, who doesn't want to hand-write the whole scikit-learn pipeline every time.
- **The pain:** building the same boilerplate over and over — profiling, cleaning, encoding, cross-validation, picking a model, evaluating honestly, explaining results.
- **What this does about it:** an LLM decides *what* to do (plan, target, feature choices), and plain deterministic Python does *all* the actual ML work. The LLM never writes or runs code.

That split — **LLM decides, tools execute** — is the core design idea. Keep it in your head; almost every design question comes back to it.

## How it works, step by step (real names)

The pipeline is a LangGraph graph. Nodes are fixed; the *order* they run in comes from a per-run plan.

1. **Upload** — `POST /api/datasets/upload` (`api/routes/datasets.py`). Checks extension and size (`max_upload_mb = 200`), then actually parses the file with `profile_dataset` before accepting it. A file that won't parse is rejected and deleted, not stored.
2. **Create run** — `POST /api/runs` → `create_run` (`api/routes/runs.py`). Seeds the state and calls `graph.invoke(...)`. **This call is synchronous** — the HTTP request blocks until the whole pipeline finishes (see Limitations).
3. **`profile_node`** — deterministic. Runs `profile_dataset`, caches `profile.json`. No LLM.
4. **`planner_node`** — LLM. `PlannerAgent` proposes an ordered list of capability names. `generate_plan` then runs `validate_plan` on it. If invalid, the validator's error text is fed back to the LLM and it retries, up to `plan_attempts = 3`. No valid plan after 3 tries → status `needs_input` (it stops for a human instead of running a bad plan).
5. **`route`** — a pure function (`planner/planner_node.py`). After every node it looks at the state and decides the next node: walk the plan by cursor, or divert to `reflect` / `planner` / stop. This is the executor.
6. **`insights_node`** *(optional)* — LLM (`EDAInsightsAgent`). Nothing depends on it; the run works fine if it's skipped.
7. **`problem_spec_node`** — LLM (`ProblemSpecAgent`). Decides `problem_type`, `target_column`, `drop_columns`. Raises `ValueError` if the LLM names a target column that doesn't exist.
8. **`cleaning_node`** — deterministic (`clean_dataset`). Drops columns, drops duplicates, writes `cleaned.parquet`.
9. **`feature_plan_node`** — LLM (`FeatureEngineeringAgent`). Important detail: it profiles the **cleaned** data, not the raw data, so skew and dtypes reflect what actually survives.
10. **`features_node`** — deterministic (`engineer_features`). Only row-wise transforms here (datetime expansion, log transforms, drops). Writes `features.parquet`.
11. **`training_node`** — deterministic (`train_models`). Cross-validates candidate models. Reads `cv_folds` from the spec, defaulting to `CV_FOLDS = 5`. Refits the winner on all data, saves `model.joblib`.
12. **`evaluation_node`** — deterministic (`evaluate_model`). Re-splits (`TEST_SIZE = 0.2`), refits on the train split, scores on the held-out split for honest metrics.
13. **`explain_node`** — deterministic (`explain_model`). SHAP if available, else permutation importance, else marks itself "unavailable". **Never raises.**
14. **`recommendations_node`** — LLM (`RecommendationsAgent`). Turns metrics + drivers into business suggestions (capped at `MAX_RECOMMENDATIONS = 6`).
15. **`report_node`** — deterministic (`assemble_report`). Writes `report.md`.
16. **`summarize_node`** — plain Python, no LLM. Builds the text summary, sets `status = "completed"`. Degrades gracefully if insights were skipped.
17. **Result** — `create_run` writes `run.json` with the whole audit trail (plan, reports, reflection history). Frontend reads it and renders the report.

### The self-fix loop (the part to really know)

Every capability node is wrapped by `_make_capability_node` (`graph/builder.py`). The wrapper **catches all exceptions** and writes `plan_error`, `failed_capability`, `failed_exc_type` into the state — it never lets the graph crash. Then:

1. `route` sees `plan_error` and sends the run to **`reflect_node`** (`reflection/node.py`).
2. `reflect_node` checks budgets first: `reflection_attempts` (2 per capability) and `repair_attempts` (4 per run). If either is spent → escalate to planner.
3. Otherwise it diagnoses: `ReflectionAgent.diagnose` runs `taxonomy.classify` (pure regex rules) first. If confidence ≥ `HEURISTIC_MIN_CONFIDENCE = 0.6`, it trusts the heuristic. If not, it falls back to an LLM diagnosis — run inside a `ThreadPoolExecutor` with a `diagnosis_timeout_s = 20.0` timeout, and on timeout/error it returns the low-confidence heuristic. **Diagnosis never raises.**
4. `select_repair` maps the diagnosis to either a concrete repair or an escalation.
5. If it's a repair, a runner from `REPAIR_REGISTRY` (9 registered) fixes the artifact — usually rewriting `features.parquet` **in place** via a pure primitive in `tools/repair_ops.py`.
6. If the repair actually changed something, `reflect_node` clears the error and sets `pending_retry_capability = <the failed step>`. `route` then re-runs that same capability. The plan cursor never moves during reflection.
7. If the repair was a no-op, or the concern is a "planning" one (model config, resource limit), it escalates to the planner instead.

Motivating example (there's a golden test for it): training fails with `could not convert string to float: 'Yes'`. The taxonomy classifies it as `TARGET_ENCODING_ERROR` (confidence 0.9) → `label_encode_target` repair → the `Yes/No` target column becomes `0/1` → training retries → succeeds.

## Text architecture diagram

```
                        ┌─────────────────────────────────────────────┐
   upload CSV/Parquet   │                 FRONTEND                      │
   + problem text  ───► │   Next.js  (hero-section.tsx -> report-view)  │
                        └───────────────┬───────────────────────────────┘
                                        │  POST /api/datasets/upload
                                        │  POST /api/runs   (blocks until done)
                        ┌───────────────▼───────────────────────────────┐
                        │              FastAPI  (api/routes)             │
                        │   datasets.py  ·  runs.py -> graph.invoke()     │
                        └───────────────┬───────────────────────────────┘
                                        │
                        ┌───────────────▼───────────────────────────────┐
                        │            LANGGRAPH  (graph/builder.py)        │
                        │   START ─► profile ─► planner ─► route ─► ...   │
                        │                                                 │
                        │   route() walks the plan one capability at a    │
                        │   time and checkpoints after every node.        │
                        └───────────────┬───────────────────────────────┘
                                        │
        ┌───────────────────────────────┼────────────────────────────────┐
        │ DECIDERS (LLM agents)          │   EXECUTORS (deterministic tools)│
        │ ─ ProblemSpecAgent             │   ─ profile_dataset              │
        │ ─ FeatureEngineeringAgent      │   ─ clean_dataset                │
        │ ─ EDAInsightsAgent (optional)  │   ─ engineer_features            │
        │ ─ RecommendationsAgent         │   ─ train_models (CV, sklearn)   │
        │ ─ PlannerAgent                 │   ─ evaluate_model (held-out)    │
        │        │                       │   ─ explain_model (SHAP/perm)    │
        │        ▼                       │   ─ assemble_report              │
        │  OpenRouterClient  ──► OpenRouter (OpenAI SDK as transport only)  │
        └────────────────────────────────┬───────────────────────────────┘
                                          │  a node raises?
                        ┌─────────────────▼──────────────────────────────┐
                        │        SELF-FIX LOOP  (reflection/)             │
                        │  wrapper catches exc -> plan_error in state     │
                        │  reflect_node:                                  │
                        │    budgets -> diagnose (taxonomy, LLM fallback) │
                        │             -> select_repair                    │
                        │             -> REPAIR_REGISTRY runner           │
                        │             -> retry same capability            │
                        │             OR escalate to planner              │
                        └─────────────────┬──────────────────────────────┘
                                          │
                        ┌─────────────────▼──────────────────────────────┐
                        │  STATE + CHECKPOINTS                            │
                        │  RunState (TypedDict): paths + summaries only   │
                        │  SqliteSaver -> checkpoints.db after every node │
                        │  dataframes / models live on disk (storage/)    │
                        └─────────────────────────────────────────────────┘
```

Two things to stress when you draw this on a whiteboard:
- **The graph is static; the plan is dynamic.** Nodes are registered once; `route` + the plan decide the path per run.
- **State holds references, not payloads.** DataFrames and models sit on disk; `RunState` carries file paths and small report dicts. That's what makes SQLite checkpointing after every node cheap.

## Tech stack — and why each piece is here (project-specific)

**Backend**
- **FastAPI** — the API surface (`/datasets`, `/runs`). Also gives typed request/response models via Pydantic for free.
- **LangGraph** — the pipeline is a graph with a conditional-edge router and a checkpointer. It's used specifically for two features this project leans on: `route`-driven dynamic ordering, and `SqliteSaver` checkpointing after every node so a run can resume.
- **Pydantic / pydantic-settings** — every LLM agent returns a Pydantic model, and the LLM is *forced* to match that schema (see engineering decisions). `Settings` reads all the knobs from env with prefix `APP_`.
- **pandas / numpy** — all the data work: profiling, cleaning, feature ops, repair primitives.
- **scikit-learn** — `Pipeline` + `ColumnTransformer` for leak-safe preprocessing, `cross_val_score`, `train_test_split`, metrics, `permutation_importance`.
- **XGBoost + LightGBM** — two of the four candidate models in both classification and regression.
- **SHAP** — model explanations. Imported behind a guard (`_HAS_SHAP`) so a missing/broken SHAP install falls back to permutation importance instead of breaking the run.
- **joblib** — saves/loads the fitted pipeline (`model.joblib`).
- **pyarrow / parquet** — intermediate artifacts (`cleaned.parquet`, `features.parquet`) are Parquet: typed, compact, fast to re-read on retry.
- **OpenAI SDK** — used *only* as the HTTP transport to OpenRouter (`base_url="https://openrouter.ai/api/v1"`). The project isn't tied to OpenAI models.

**Frontend**
- **Next.js 14 / React 18** — the upload + report UI. `hero-section.tsx` does the upload and run; `report-view.tsx` renders the leaderboard, held-out metrics, SHAP bars, recommendations, and download buttons.
- **framer-motion / three.js / tailwind / lucide-react** — presentation (animation, 3D hero, styling, icons). Not central to the ML story.

## Genuinely interesting engineering decisions

1. **LLM plans, but only from a fixed menu — and a validator checks it before anything runs.** The planner can only emit names that exist in `REGISTRY` (11 capabilities). `validate_plan` then checks the plan as a DAG: names are registered, each step's `requires` are satisfied by earlier `produces`, no artifact is produced twice, and the terminal artifacts (`summary`, `report`) are reached. A bad plan is rejected and the error is fed back to the LLM. So the LLM's freedom is boxed in on both sides — it can't invent a step, and it can't emit an unsound order.

2. **Universal structured output without relying on a provider's JSON mode.** `OpenRouterClient.complete(system, user, schema)` embeds `schema.model_json_schema()` in the prompt, asks for a single JSON object, strips code fences, and validates with `schema.model_validate_json`. On a `ValidationError` it feeds the error back and retries up to `MAX_VALIDATION_RETRIES = 2`. This works across any model behind OpenRouter, not just ones with native structured output.

3. **The self-fix layer separates "fix the data" from "change the plan," and never confuses the two.** `select_repair` returns exactly one of: a `RepairSpec` (rewrite an artifact, retry the same step) or an escalation (hand to the planner). Data/encoding/NaN/dtype problems get repaired and retried. Model-config and resource problems escalate. That boundary is enforced in one place, which is why the loop is easy to reason about.

4. **Diagnosis is heuristic-first, LLM-second, and time-boxed.** `taxonomy.classify` is pure regex over exception type + message — fast, free, deterministic. The LLM is only consulted when the heuristic is unsure (< 0.6 confidence), and even then it runs under a 20-second timeout with the heuristic as the fallback. So the common cases cost nothing and the graph can't hang on a slow model.

5. **Leak-safe by construction.** The `features_node` tool only does row-wise transforms. Everything that *learns a parameter* — imputation values, scaling stats, one-hot categories — lives inside the sklearn `Pipeline`, so it's fitted per fold during cross-validation. There's no separate "fit the scaler on all data then split" step to leak information.

## Honest limitations (what's actually in the code)

- **`graph.invoke()` is synchronous and blocking** (`runs.py:106`). The HTTP request for `POST /api/runs` stays open for the entire pipeline — profiling, training, the lot. There's no job queue or background worker. This is the first thing that breaks under real load, and it's the honest answer to "how does this scale?"
- **Planner escalation doesn't actually change strategy yet.** When reflection can't fix something, it escalates to the planner. But `_handle_escalation` today just **retries the same plan from the same failed step**, and gives up to `needs_input` after `replan_attempts = 2`. The `planner_hint` (e.g. "swap the model", "reduce search space") is recorded in warnings but **not acted on**. The code says this plainly — it's a deliberate seam for future work, not a bug. Don't oversell the escalation branch as an intelligent replan; it's currently a graceful give-up.
- **`validate_schema` repair has a chicken-and-egg limit.** It realigns feature columns to `training_report.feature_columns` — which only exists if training already succeeded once. On a *first-ever* training failure there's nothing to align to, so it no-ops and the layer escalates. It degrades cleanly (doesn't crash), but it can't help the case you'd most expect.
- **Single-process, single-machine.** State checkpoints go to a local SQLite file (`checkpoints.db`) and artifacts to the local filesystem (`storage/`). Nothing is shared across machines. Fine for one box; not built for horizontal scaling as-is.
- **End-to-end self-heal is tested for one path, not all.** `test_router_resume.py` drives the *full compiled graph* through a failure → reflect → repair → retry → `completed` — but that path uses the `retry_llm` (transient) repair. The artifact-rewriting repairs (`label_encode_target`, etc.) are proven by node-level tests (`test_reflect_node.py`) and pure-function unit tests (`test_repair_ops.py`), not driven through the whole graph. So the *mechanism* is end-to-end tested; each individual data repair is unit-tested.
- **A knob value rides inside a human-readable string.** `reduce_cv_folds` returns `detail="reduce cv_folds 5 -> 4"`, and `reflect_node._parse_trailing_int` pulls the new fold count off the *end* of that string. It works and is tested, but the wording of that detail string is effectively a contract — a slightly fragile coupling.
- **LLM guardrails catch nonsense, not bad judgment.** Agents are told "never invent numbers/columns," outputs are schema-validated, and `problem_spec` raises if the target column doesn't exist. That catches hallucinated/nonexistent columns. It does **not** catch a *plausible but wrong* choice among real columns.
- **Frontend: I traced the main path only.** `page.tsx`, `hero-section.tsx`, and `report-view.tsx` are the real working flow. Other section components exist in the repo; I did not fully verify how (or whether) they're wired in.
- **README is slightly behind the code.** It lists the frontend as "pending" (it exists now) and the `repairs.py` comments mention "Phase E" as future work, but the `cv_folds` knob is in fact fully wired (training reads it, reflect persists it, a test confirms 5→4). Minor doc drift worth knowing before an interviewer catches it.

---

# Section 2 — Interview Questions (with one-line hints)

> Hints point you at the right answer; they aren't the full answer. Practice saying each one out loud.

## 1. Walk me through it

1. "Walk me through what happens from upload to final report." — *Hint: upload validates by parsing → `create_run` → profile → planner+validator → route walks the plan → deterministic tools + LLM agents → summarize sets `completed`.*
2. "Where does the LLM actually get used, and where doesn't it?" — *Hint: LLM in planner + 4 agents (problem/feature/insights/recommendations); everything else is deterministic tools; LLM never runs code.*
3. "What decides the order the steps run in?" — *Hint: `PlannerAgent` proposes names, `validate_plan` checks the DAG, `route` walks the plan by cursor.*
4. "Show me the single most interesting flow in the system." — *Hint: the `Yes/No` target failure → taxonomy `TARGET_ENCODING_ERROR` → `label_encode_target` → retry → success.*
5. "What does the user get at the end?" — *Hint: `run.json` (plan, reports, reflection history) + downloadable `model.joblib` and `report.md`.*

## 2. Architecture & design

1. "Why LangGraph and not a plain function pipeline or Celery chain?" — *Hint: conditional-edge routing for dynamic order + retry loops, and per-node checkpointing for resume.*
2. "Why is the graph static but the plan dynamic?" — *Hint: nodes registered once in `build_graph` (`@lru_cache`); `route` + the plan pick the path per run.*
3. "Why does state hold file paths instead of the DataFrames themselves?" — *Hint: keeps `RunState` and every checkpoint small; heavy data lives on disk as Parquet/joblib.*
4. "How do you stop the LLM from producing an impossible plan?" — *Hint: registry-bounded names + `validate_plan` DAG check (requires/produces/duplicates/terminal) + feedback-retry.*
5. "Why separate 'repair the artifact' from 'escalate to the planner'?" — *Hint: data problems are retryable in place; strategy problems need a different plan; `select_repair` enforces one-or-the-other.*
6. "How does structured output work without a provider JSON mode?" — *Hint: schema-in-prompt + `model_validate_json` + bounded validation-retry in `OpenRouterClient`.*
7. "Why is the OpenAI SDK here if you're not using OpenAI?" — *Hint: it's only the transport to OpenRouter's OpenAI-compatible endpoint; the abstraction is `LLMClient`.*

## 3. Deep-dive code

1. "What exactly happens when a capability node raises?" — *Hint: `_make_capability_node` catches it, writes `plan_error`/`failed_capability`/`failed_exc_type`, never re-raises.*
2. "Walk me through `reflect_node`'s decision order." — *Hint: check budgets → diagnose → `select_repair` → run repair → if changed retry, else escalate; cursor never advances.*
3. "How does diagnosis avoid hanging on a slow LLM?" — *Hint: heuristic first (≥ 0.6 trust); LLM fallback in a `ThreadPoolExecutor` under `diagnosis_timeout_s = 20.0`; returns heuristic on timeout.*
4. "How is data leakage prevented during training?" — *Hint: only row-wise transforms in `features_node`; impute/scale/encode live inside the sklearn `Pipeline`, fitted per fold.*
5. "How are the held-out metrics computed honestly?" — *Hint: `evaluate_model` re-splits (`TEST_SIZE=0.2`), `clone`s the pipeline, refits on train only, scores on the held-out split.*
6. "How does `explain_model` guarantee it never breaks the run?" — *Hint: guarded `import shap`; TreeExplainer/LinearExplainer → permutation fallback → method `"unavailable"`; never raises.*
7. "How does the `cv_folds` knob repair actually reach the training tool?" — *Hint: `reduce_cv_folds` sets `artifact_key="cv_folds"`; `reflect_node` deep-copies it into `problem_spec`; `training_node` reads `spec.get("cv_folds")`.*

## 4. Why this, not that (trade-offs)

1. "Why let deterministic tools do the ML instead of asking the LLM to write code?" — *Hint: reproducible, testable, no arbitrary code execution; LLM decisions are cheap to validate.*
2. "Why heuristic-first diagnosis instead of always asking the LLM?" — *Hint: common failures are cheap/deterministic/offline; LLM only for the uncertain tail.*
3. "Why SQLite checkpointing instead of just re-running on failure?" — *Hint: resume from the last good node; don't redo expensive training; but it's single-machine (trade-off).*
4. "Why Parquet for intermediates instead of CSV or in-memory?" — *Hint: typed + compact + fast re-reads on retry; repairs rewrite it in place at the same path.*
5. "Why cap recommendations at 6 and retries at small numbers (3/2/2/4)?" — *Hint: bounded budgets guarantee termination; `recursion_limit` sized above them as the final backstop.*

## 5. Failure & edge cases (checked against the code)

1. "A user uploads a file that isn't really a CSV — what happens?" — *Hint: `datasets.py` parses with `profile_dataset` before accepting; parse failure → rejected and deleted, no run created.*
2. "The target column is `Yes/No` strings and training crashes — then what?" — *Hint: caught → `TARGET_ENCODING_ERROR` → `label_encode_target` rewrites to 0/1 → same step retried → completes.*
3. "A repair runs but changes nothing — does it keep retrying forever?" — *Hint: no; `applied && changed` gate — a no-op escalates instead of burning a retry; `test_noop_repair_escalates_without_retry`.*
4. "Reflection can't fix it — is the failure ever intelligently re-planned?" — *Hint: honestly, no yet — `_handle_escalation` retries the same plan then stops at `needs_input`; the hint is recorded, not acted on.*
5. "The planner LLM keeps emitting an invalid plan — does the request hang or 500?" — *Hint: neither; after `plan_attempts=3` it returns `needs_input` with the validator errors; a node exception becomes a `failed` `run.json`, not HTTP 500.*

## 6. Scaling & production

1. "This blocks the HTTP request for the whole run — how would you fix that?" — *Hint: background worker / job queue; `POST /runs` returns an id immediately; poll `GET /runs/{id}`; the run record already models this.*
2. "How would you run this across multiple machines?" — *Hint: today it's single-process (SQLite checkpoints + local `storage/`); move checkpoints to a shared DB and artifacts to object storage.*
3. "A dataset is 5 GB — where does it fall over, and what changes?" — *Hint: everything loads into pandas in-memory; need chunking/out-of-core or a sampling step; `max_upload_mb=200` caps it today.*
4. "How would you observe this in production?" — *Hint: `reflection_history` is already an audit trail; add metrics on repair frequency, escalation rate, per-node latency.*
5. "How would you make the escalation branch actually useful?" — *Hint: act on `planner_hint` — model swap, shrink search space, edit the plan — the seam is already there in `_handle_escalation`.*

## 7. Reflection

1. "What's the part you're most proud of, and why?" — *Hint: the diagnose→repair→retry loop with a clean repair-vs-escalate boundary and bounded budgets.*
2. "What would you rebuild if you started over?" — *Hint: async runs from day one; and make escalation act on the hint instead of just retrying.*
3. "What surprised you or was harder than expected?" — *Hint: keeping the self-fix loop from looping — the three budgets + `recursion_limit` backstop exist for exactly that.*
4. "If you had one more week, what's the highest-value thing to add?" — *Hint: background job queue (unblocks the API) or a real replan on escalation; pick one and justify it.*
