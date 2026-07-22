"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import datasets, insights, runs
from app.core.config import settings

app = FastAPI(
    title="AI Data Science Platform API",
    description="Autonomous data science workflows powered by AI agents",
    version="0.1.0",
)

# Open CORS for local development so any frontend can consume the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(insights.router, prefix="/api/datasets", tags=["insights"])
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
