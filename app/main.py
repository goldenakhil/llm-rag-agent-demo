import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from .rag import RagPipeline
from .react_agent import ReactAgent

load_dotenv()  # loads .env if present


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    steps: List[dict]
    retrieved_chunks: List[dict]


def create_app() -> FastAPI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")

    client = OpenAI(api_key=api_key)

    rag = RagPipeline.from_local_docs(data_dir="data", client=client, max_chunks=3)
    agent = ReactAgent(client=client, rag=rag, max_steps=3)

    app = FastAPI(title="RAG + ReAct Agent API")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        result = agent.run(req.query)
        return QueryResponse(
            answer=result["answer"],
            steps=result["steps"],
            retrieved_chunks=result["retrieved_chunks"],
        )

    return app


app = create_app()
