import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.question_officer import QuestionOfficerRequest, QuestionOfficerResponse

CONTRACT_DIR = Path(__file__).resolve().parents[4] / "docs" / "server" / "contracts"

def _load(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))

def test_request_sample_passes():
    req = QuestionOfficerRequest.model_validate(_load("question_officer_req.json"))
    assert req.direction_default is None   # 2026-11-08 四级语义链：None = Java 未声明
    assert req.history_digest == ["第1轮: Java内存模型相关2题"]
    assert req.resume_url.startswith("http://minio:9000/")

def test_success_response_sample_passes():
    resp = QuestionOfficerResponse.model_validate(_load("question_officer_resp.json"))
    assert resp.intent == "GENERATE"
    assert resp.questions_source == "kb+resume"
    assert resp.resume_score == 78
    assert resp.small_talk is None
    assert len(resp.questions) == 10

def test_smalltalk_response_sample_passes():
    resp = QuestionOfficerResponse.model_validate(_load("question_officer_smalltalk.json"))
    assert resp.intent == "SMALL_TALK"
    assert resp.questions == []
    assert resp.resume_score is None

def test_missing_required_field_rejected():
    """trace_id 必填：缺失必须被拒 —— 这就是线上 VALIDATION_ERROR=422 的来源。"""
    broken = _load("question_officer_req.json")
    del broken["trace_id"]
    with pytest.raises(ValidationError):
        QuestionOfficerResponse.model_validate(broken)