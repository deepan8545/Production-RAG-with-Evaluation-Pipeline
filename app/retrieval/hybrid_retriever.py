"""
Hybrid retriever
----------------
Combines BM25 (keyword) + Neo4j vector (semantic) search via
Reciprocal Rank Fusion (RRF). Returns top-K fused candidates.
"""
import pickle
from pathlib import Path

from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document
from loguru import logger

from app.config import settings
from app.ingestion.embeddings import get_embeddings


BM25_INDEX_PATH = Path("data/bm25_index.pkl")
DOCS_CACHE_PATH = Path("data/docs_cache.pkl")


def _load_bm25():
    """Load persisted BM25 index and document cache."""
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            "BM25 index not found. Run ingestion first:\n"
            "  python -m app.ingestion.ingest"
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(DOCS_CACHE_PATH, "rb") as f:
        docs = pickle.load(f)
    return bm25, docs


def _load_neo4j_retriever():
    """Connect to existing Neo4j vector index."""
    embeddings = get_embeddings()
    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        index_name="document_chunks",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    return vector_store.as_retriever(
        search_kwargs={"k": settings.retriever_top_k}
    )


def reciprocal_rank_fusion(
    bm25_docs: list[Document],
    vector_docs: list[Document],
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    k: int = 60,  # RRF constant — larger = less steep rank decay
) -> list[Document]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Score = bm25_weight * (1 / (k + rank)) + vector_weight * (1 / (k + rank))
    Returns docs sorted by combined score, deduplicated by chunk_id.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight * (1 / (k + rank + 1))
        doc_map[doc_id] = doc

    for rank, doc in enumerate(vector_docs):
        doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight * (1 / (k + rank + 1))
        doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


class HybridRetriever:
    """
    Single object that owns both retrievers and exposes a
    get_relevant_documents(query) method compatible with LangChain chains.
    """

    def __init__(self):
        logger.info("Initialising hybrid retriever...")
        self._bm25, self._docs = _load_bm25()
        self._vector_retriever = _load_neo4j_retriever()
        logger.info("Hybrid retriever ready")

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in top_indices]

    def get_relevant_documents(self, query: str) -> list[Document]:
        k = settings.retriever_top_k

        bm25_results = self._bm25_search(query, k)
        vector_results = self._vector_retriever.get_relevant_documents(query)

        fused = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            bm25_weight=settings.bm25_weight,
            vector_weight=settings.vector_weight,
        )

        logger.debug(
            f"Retrieval | query='{query[:60]}' | "
            f"bm25={len(bm25_results)} vector={len(vector_results)} fused={len(fused)}"
        )
        return fused[: settings.retriever_top_k]

    # Make it work as a LangChain Runnable
    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)
