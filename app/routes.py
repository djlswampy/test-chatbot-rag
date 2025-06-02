from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chat import ask

router = APIRouter()

# 클라이언트에서 전달받는 POST 요청의 JSON 데이터 구조를 정의한 클래스
class ChatRequest(BaseModel):
    question: str

# 라우트 함수(엔드포인트 핸들러)
# /chat 경로로 들어오는 클라이언트의 POST 요청을 받을 수 있음
@router.post("/chat")
async def chat(req: ChatRequest):
    answer = ask(req.question)
    return {"answer": answer}