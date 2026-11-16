"""
评分官：对单题作答打分，产出线上契约
本文件是唯一知道评分流程的地方（对应 ADR-006 的“复制模式”）
"""
import time
from venv import logger

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.llm import structured_call, Purpose
from app.core.prompts import EVALUATOR_SYSTEM_PROMPT
from app.core.trace import write_span, get_trace_id
from app.schemas.evaluation import EvaluationRequest, EvaluationResponse, EvaluationDraft


def _user_content(req: EvaluationRequest) -> str:
    """把请求拼成用户消息：有参考要点就用，没有就让模型自行判断"""
    lines = [f"【题目】{req.question}", f"【候选人作答】{req.answer}"]
    if req.resume_digest:
        lines.append("【简历要点】" + "；".join(req.resume_digest))
    if req.standard:
        lines.append("【参考要点】" + "；".join(req.standard))
    return "\n".join(lines)

def run_evaluator(req: EvaluationRequest) -> EvaluationResponse:
    """单题评分：结构化输出 → 一致性兜底 → 回填 trace_id"""
    start = time.perf_counter()

    # step1 结构化打分（Purpose.SCORE 固定 temperature=0.2，满足一致性要求）
    draft: EvaluationDraft = structured_call(
        schema=EvaluationDraft,
        messages=[
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=_user_content(req))
        ],
        purpose=Purpose.SCORE
    )

    # step2 契约一致性（降级策略）：说要追问却没给问题，降级为不追问
    if draft.follow_up_needed and not draft.follow_up_question:
        logger.warning("follow_up_needed=true 但缺少追问问题，已降级为 false: trace={}", req.trace_id)
        draft.follow_up_needed = False

    # step3 观测：写一个 span，便于后续 trace 页面按 trace_id 串起来
    duration_ms = round((time.perf_counter() - start) * 1000)
    write_span(
        "evaluator",
        trace_id = req.trace_id or get_trace_id(),
        score = draft.score,
        follow_up_needed = draft.follow_up_needed,
        duration_ms = duration_ms,
    )

    return EvaluationResponse(
        trace_id=req.trace_id,
        score=draft.score,
        feedback=draft.feedback,
        missing_points=draft.missing_points,
        follow_up_needed=draft.follow_up_needed,
        follow_up_question=draft.follow_up_question,
    )
