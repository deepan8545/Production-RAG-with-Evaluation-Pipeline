"""
FastAPI application
-------------------
Endpoints:
  GET  /health          — liveness check
  POST /query           — full RAG pipeline (retrieve → rerank → generate)
  POST /ingest          — trigger ingestion from a directory path
  GET  /docs            — auto-generated Swagger UI
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.config import settings
from app.generation.generator import generate_answer
from app.ingestion.ingest import run_ingestion
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import rerank


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None   # override RERANKER_TOP_K per-request if needed


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    sources: list[str]
    chunks_retrieved: int
    chunks_after_rerank: int


class IngestRequest(BaseModel):
    docs_dir: str = "data/documents"


class HealthResponse(BaseModel):
    status: str
    model: str
    neo4j_uri: str


# ── App lifecycle ─────────────────────────────────────────────────────────────

retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Starting up RAG service...")
    try:
        retriever = HybridRetriever()
        logger.info("Retriever initialised successfully")
    except FileNotFoundError as e:
        logger.warning(f"Retriever not ready (run ingestion first): {e}")
        retriever = None
    yield
    logger.info("Shutting down RAG service")


app = FastAPI(
    title="Production RAG API",
    description="Hybrid retrieval (BM25 + Neo4j vector) with cross-encoder reranking and citation-grounded generation via Claude.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if retriever else "retriever_not_ready",
        model=settings.claude_model,
        neo4j_uri=settings.neo4j_uri,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not retriever:
        raise HTTPException(
            status_code=503,
            detail="Retriever not ready. Run ingestion first: python -m app.ingestion.ingest",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Hybrid retrieval
    candidates = retriever.get_relevant_documents(request.question)

    # 2. Cross-encoder reranking
    if request.top_k:
        import app.config as cfg
        cfg.settings.reranker_top_k = request.top_k

    reranked = rerank(request.question, candidates)

    # 3. Citation-grounded generation
    result = generate_answer(request.question, reranked)

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        citations=result["citations"],
        sources=result["sources"],
        chunks_retrieved=len(candidates),
        chunks_after_rerank=len(reranked),
    )


@app.post("/ingest")
async def ingest(request: IngestRequest):
    docs_path = Path(request.docs_dir)
    if not docs_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Directory not found: {request.docs_dir}",
        )
    try:
        run_ingestion(request.docs_dir)
        global retriever
        retriever = HybridRetriever()
        return {"status": "ingestion complete", "docs_dir": request.docs_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
