from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

import ai_inference as ai
from llm_queue import submit, CHAT
from one_ring import start_scheduler

@asynccontextmanager
async def lifespan(app):
    sched = start_scheduler()
    yield
    sched.shutdown(wait=False)

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/chat")
def chat(req: ChatRequest):          # plain def, not async
    return {"answer": submit(CHAT, ai.chat_job, req.session_id, req.question).result()}

@app.get("/notifications")
def notifications():
    return {"rows": ai.read_results()}