from fastapi import APIRouter
from schemas import ChatRequest
from agent import run_agent

router = APIRouter()


@router.post("/agent/chat")

async def chat(request: ChatRequest):

    response = run_agent(request.message)

    return {"response": response}