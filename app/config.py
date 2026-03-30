from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    claude_model: str = Field("claude-3-5-sonnet-20241022", env="CLAUDE_MODEL")
    max_tokens: int = Field(1024, env="MAX_TOKENS")
    temperature: float = Field(0.0, env="TEMPERATURE")

    # Embeddings
    embedding_provider: str = Field("openai", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")

    # Neo4j
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field("password123", env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")

    # Retrieval
    retriever_top_k: int = Field(20, env="RETRIEVER_TOP_K")
    reranker_top_k: int = Field(5, env="RERANKER_TOP_K")
    bm25_weight: float = Field(0.4, env="BM25_WEIGHT")
    vector_weight: float = Field(0.6, env="VECTOR_WEIGHT")

    # RAGAS thresholds
    faithfulness_threshold: float = Field(0.85, env="FAITHFULNESS_THRESHOLD")
    answer_relevancy_threshold: float = Field(0.80, env="ANSWER_RELEVANCY_THRESHOLD")
    context_precision_threshold: float = Field(0.75, env="CONTEXT_PRECISION_THRESHOLD")

    # App
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
