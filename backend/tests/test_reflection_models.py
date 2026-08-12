"""Tests for the reflection layer's Pydantic models (Phase A)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.reflection.models import (
    Diagnosis,
    FailureCategory,
    PlannerHint,
    ReflectionRecord,
    RepairResult,
    RepairSpec,
)


def test_failure_category_is_string_enum_with_full_taxonomy():
    # The fixed taxonomy from the design — every member present, string-valued.
    expected = {
        "DATA_SCHEMA_ERROR",
        "TARGET_ENCODING_ERROR",
        "MISSING_VALUES",
        "TYPE_ERROR",
        "MODEL_CONFIGURATION_ERROR",
        "RESOURCE_LIMIT",
        "TRANSIENT_ERROR",
        "VALIDATION_ERROR",
        "UNKNOWN_ERROR",
    }
    assert {c.value for c in FailureCategory} == expected
    # str-enum: value equals its own string, so it serialises transparently.
    assert FailureCategory.TARGET_ENCODING_ERROR == "TARGET_ENCODING_ERROR"


def test_diagnosis_defaults_planner_hint_and_source():
    d = Diagnosis(
        category=FailureCategory.MISSING_VALUES,
        confidence=0.85,
        reason="NaNs present",
    )
    assert d.planner_hint is None
    assert d.suggested_repair is None
    assert d.evidence == {}
    assert d.source == "heuristic"


def test_diagnosis_confidence_is_bounded():
    with pytest.raises(ValidationError):
        Diagnosis(category=FailureCategory.UNKNOWN_ERROR, confidence=1.5, reason="x")
    with pytest.raises(ValidationError):
        Diagnosis(category=FailureCategory.UNKNOWN_ERROR, confidence=-0.1, reason="x")


def test_diagnosis_roundtrips_through_model_dump():
    # Stored into RunState via model_dump(), rehydrated on read.
    d = Diagnosis(
        category=FailureCategory.TARGET_ENCODING_ERROR,
        confidence=0.9,
        reason="non-numeric target",
        suggested_repair="label_encode_target",
        planner_hint=None,
        evidence={"offending_value": "Yes"},
    )
    dumped = d.model_dump()
    restored = Diagnosis.model_validate(dumped)
    assert restored == d
    assert restored.evidence["offending_value"] == "Yes"


def test_planner_hint_defaults_are_inert():
    h = PlannerHint()
    assert h.requires_model_swap is False
    assert h.reduce_search_space is False
    assert h.note == ""


def test_repair_result_carries_change_and_error_flags():
    ok = RepairResult(
        repair_name="label_encode_target",
        applied=True,
        changed=True,
        detail="encoded Yes/No -> 1/0",
        artifact_key="feature_report",
    )
    assert ok.error is None

    failed = RepairResult(
        repair_name="coerce_numeric",
        applied=False,
        changed=False,
        detail="nothing to coerce",
        error="column not found",
    )
    assert failed.applied is False
    assert failed.error == "column not found"


def test_reflection_record_roundtrips_with_nested_models():
    record = ReflectionRecord(
        failed_capability="training",
        original_error="could not convert string to float: 'Yes'",
        diagnosis=Diagnosis(
            category=FailureCategory.TARGET_ENCODING_ERROR,
            confidence=0.9,
            reason="non-numeric target",
            suggested_repair="label_encode_target",
        ),
        repair=RepairSpec(
            name="label_encode_target", target_artifact_key="feature_report"
        ),
        repair_result=RepairResult(
            repair_name="label_encode_target",
            applied=True,
            changed=True,
            detail="encoded",
        ),
        retry_succeeded=True,
        attempt=1,
        duration_ms=42,
    )
    restored = ReflectionRecord.model_validate(record.model_dump())
    assert restored == record
    assert restored.diagnosis.category is FailureCategory.TARGET_ENCODING_ERROR


def test_reflection_record_allows_no_repair():
    # A diagnosis that escalates straight to the planner has no repair attached.
    record = ReflectionRecord(
        failed_capability="training",
        original_error="out of memory",
        diagnosis=Diagnosis(
            category=FailureCategory.RESOURCE_LIMIT,
            confidence=0.8,
            reason="OOM",
            planner_hint=PlannerHint(reduce_search_space=True),
        ),
        attempt=1,
        duration_ms=5,
    )
    assert record.repair is None
    assert record.repair_result is None
    assert record.retry_succeeded is None
