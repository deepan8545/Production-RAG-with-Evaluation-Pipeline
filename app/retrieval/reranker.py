"""
Cross-encoder reranker
----------------------
Takes the hybrid retriever's top-K candidates and rescores each
(query, chunk) pair using a cross-encoder model — much more accurate
than bi-encoder similarity alone. Only run on the small candidate set.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, free, local)
"""
from functools import lru_cache

from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    logger.info("Loading cross-encoder model (first load downloads ~80MB)...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    logger.info("Cross-encoder loaded")
    return model


def rerank(query: str, candidates: list[Document]) -> list[Document]:
    """
    Rescore candidates with the cross-encoder and return top RERANKER_TOP_K.
    """
    if not candidates:
        return []

    model = _load_cross_encoder()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    top_k = settings.reranker_top_k
    reranked = [doc for doc, _ in scored[:top_k]]

    logger.debug(
        f"Reranker | {len(candidates)} candidates → {len(reranked)} chunks "
        f"| top score: {scored[0][1]:.3f}"
    )
    return reranked
