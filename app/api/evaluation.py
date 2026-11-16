"""评分官接口：POST /v1/agents/evaluation/run"""
from fastapi import APIRouter
from fastapi.params import Depends

from app.agents.evaluator.evaluator import run_evaluator
from app.api.deps import require_token
from app.schemas.evaluation import EvaluationResponse, EvaluationRequest

router = APIRouter(dependencies=[Depends(require_token)])

@router.post("/agents/evaluation/run", response_model=EvaluationResponse)
def run(req: EvaluationRequest) -> EvaluationResponse:
    """def 而非 async：全链阻塞 IO（LLM 调用），由 FastAPI 丢线程池执行。"""
    return run_evaluator(req)