"""出题官接口：POST /v1/agents/question-officer/run"""
from fastapi import APIRouter, Depends

from app.agents.question_officer.officer import run_question_officer
from app.api.deps import require_token
from app.schemas.question_officer import QuestionOfficerResponse, QuestionOfficerRequest

router = APIRouter(dependencies=[Depends(require_token)])

@router.post("/agents/question-officer/run", response_model=QuestionOfficerResponse)
def run(req: QuestionOfficerRequest) -> QuestionOfficerResponse:
    """def 而非 async：全链阻塞 IO（LLM/下载/检索），同步函数由 FastAPI 丢线程池。"""
    return run_question_officer(req)