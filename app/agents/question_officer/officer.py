"""
出题官统一装配：按流程图串起五步，产出线上契约
本文件是唯一知道全流程的地方
"""
import time

from loguru import logger

from app.agents.question_officer.direction import normalize_direction
from app.agents.question_officer.intent import classify_intent
from app.agents.question_officer.resume import ResumeAnalysis, analyze_resume
from app.agents.question_officer.retrieval import retrieve_questions
from app.agents.question_officer.smalltalk import fallback_chat
from app.agents.question_officer.synthesize import synthesize_questions
from app.core.trace import write_span, get_trace_id
from app.schemas.question_officer import QuestionOfficerRequest, QuestionOfficerResponse


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)

def run_question_officer(req: QuestionOfficerRequest) -> QuestionOfficerResponse:
    """全链路：意图 → (简历评分) → 方向 → 检索 → 合成；闲聊走兜底分支。"""
    trace: dict = {}

    #step1 意图路由
    t0 = time.perf_counter()
    intent = classify_intent(req.user_input)
    trace["intent_ms"] = _ms(t0)

    if intent.intent == "SMALL_TALK":
        # step6 兜底闲聊：questions 空、无评分，questions_source=none
        t6 = time.perf_counter()
        reply = fallback_chat(req.user_input)
        trace["chat_ms"] = _ms(t6)
        return QuestionOfficerResponse(
            trace_id=req.trace_id, intent="SMALL_TALK",
            questions=[], resume_score=None, resume_suggest=[],
            resume_type=None, questions_source="none",
            small_talk=reply, steps_trace=trace,
        )

    #step2 简历解析评分（可选：无 resume_url 跳过）
    analysis: ResumeAnalysis | None = None
    if req.resume_url:
        t2 = time.perf_counter()
        analysis = analyze_resume(req.resume_url)
        trace["resume_ms"] = _ms(t2)

    # step3 方向归一（有简历用 resume_type，无简历用 direction_default）
    direction = normalize_direction(
        resume_type=analysis.resume_type if analysis else None,
        direction_default=req.direction_default,
        user_input=req.user_input,
    )

    # step4 检索（有简历用 rag_query，无简历空列表 → retrieval 内部用 direction 兜底）
    t4 = time.perf_counter()
    hits = retrieve_questions(
        rag_query=analysis.rag_query if analysis else [],
        direction=direction,
    )
    trace["retrieve_ms"] = _ms(t4)

    # step5 合成（kb 片段 + 简历原文 + 历史摘要）
    t5 = time.perf_counter()
    questions = synthesize_questions(
        kb_hits=hits,
        resume_text=analysis.resume_text if analysis else None,
        direction=direction,
        history_digest=req.history_digest,
    )
    trace["synthesize_ms"] = _ms(t5)

    # questions_source 语义:
    # kb+resume = kb 检索命中（无论有无简历）；resume_only = kb 未命中但有简历；none = 无外部材料（闲聊；或无简历+未命中的自由生成）
    if hits:
        source = "kb+resume"
    elif analysis:
        source = "resume_only"
    else:
        source = "none"

    write_span("agent.question_officer", trace_id=get_trace_id(),
        intent="GENERATE", count=len(questions), source=source,
        steps=trace)
    logger.info(f"officer done source={source} count={len(questions)} trace={trace}")

    return QuestionOfficerResponse(
        trace_id=req.trace_id, intent="GENERATE",
        questions=questions,
        resume_score=analysis.score if analysis else None,
        resume_suggest=analysis.resume_suggest if analysis else [],
        resume_type=analysis.resume_type if analysis else None,
        questions_source=source, small_talk=None,
        steps_trace=trace,
    )