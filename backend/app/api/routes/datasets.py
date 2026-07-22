"""Datasets API — upload, list, profile.

Storage is filesystem-backed: each dataset gets its own folder with the
raw file, a metadata.json, and a pre-computed profile.json. A database
comes in only when multi-user run tracking requires it.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas.datasets import (
    DatasetListItem,
    DatasetProfileResponse,
    DatasetUploadResponse,
)
from app.core.config import settings
from app.tools.profiler import profile_dataset

router = APIRouter()

ALLOWED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}


def _dataset_dir(dataset_id: str) -> Path:
    return settings.datasets_dir / dataset_id


def _load_meta(dataset_id: str) -> dict:
    meta_path = _dataset_dir(dataset_id) / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return json.loads(meta_path.read_text())


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile) -> DatasetUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    dataset_id = uuid.uuid4().hex[:12]
    ds_dir = _dataset_dir(dataset_id)
    ds_dir.mkdir(parents=True, exist_ok=True)

    raw_path = ds_dir / f"raw{ext}"
    raw_path.write_bytes(content)

    # Validate it actually parses before accepting
    try:
        profile = profile_dataset(raw_path)
    except Exception as exc:  # parse failure → reject upload
        raw_path.unlink(missing_ok=True)
        ds_dir.rmdir()
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}") from exc

    uploaded_at = datetime.now(timezone.utc)
    meta = {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "uploaded_at": uploaded_at.isoformat(),
        "raw_path": str(raw_path),
    }
    (ds_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (ds_dir / "profile.json").write_text(json.dumps(profile, indent=2))

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        size_bytes=len(content),
        uploaded_at=uploaded_at,
    )


@router.get("", response_model=list[DatasetListItem])
def list_datasets() -> list[DatasetListItem]:
    items: list[DatasetListItem] = []
    for ds_dir in sorted(settings.datasets_dir.iterdir()):
        meta_path = ds_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            items.append(DatasetListItem(**{k: meta[k] for k in ("dataset_id", "filename", "size_bytes", "uploaded_at")}))
    return items


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
def get_profile(dataset_id: str) -> DatasetProfileResponse:
    _load_meta(dataset_id)  # 404 if missing
    profile_path = _dataset_dir(dataset_id) / "profile.json"
    profile = json.loads(profile_path.read_text())
    return DatasetProfileResponse(dataset_id=dataset_id, profile=profile)
