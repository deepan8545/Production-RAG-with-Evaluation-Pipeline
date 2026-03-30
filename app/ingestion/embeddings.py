"""
Embeddings factory.
Supports OpenAI (default) or HuggingFace sentence-transformers (free, local).

To use HuggingFace set in .env:
    EMBEDDING_PROVIDER=huggingface
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
"""
from functools import lru_cache

from loguru import logger

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    if settings.embedding_provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info(f"Using HuggingFace embeddings: {settings.embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Default: OpenAI
    from langchain_openai import OpenAIEmbeddings
    logger.info(f"Using OpenAI embeddings: {settings.embedding_model}")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
