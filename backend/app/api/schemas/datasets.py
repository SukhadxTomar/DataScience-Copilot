"""Pydantic schemas for the datasets API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    profile: dict[str, Any]


class DatasetListItem(BaseModel):
    dataset_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime
