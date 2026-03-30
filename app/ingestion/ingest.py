"""
Ingestion pipeline
------------------
Loads documents → chunks → embeds → stores in Neo4j (vector + BM25 index).

Usage:
    python -m app.ingestion.ingest --docs-dir data/documents
"""
import argparse
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_neo4j import Neo4jVector
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from loguru import logger

from app.config import settings
from app.ingestion.embeddings import get_embeddings

BM25_INDEX_PATH = Path("data/bm25_index.pkl")
DOCS_CACHE_PATH = Path("data/docs_cache.pkl")


def _load_pdf_documents(docs_dir: str) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(docs_dir).rglob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                logger.warning(f"No extractable text in {path}")
            docs.append(Document(page_content=text, metadata={"source": str(path)}))
        except Exception as e:
            logger.warning(f"Failed to read PDF {path}: {e}")
    return docs


def load_documents(docs_dir: str) -> list[Document]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    all_docs: list[Document] = []

    pdf_docs = _load_pdf_documents(docs_dir)
    all_docs.extend(pdf_docs)
    logger.info(f"Loaded {len(pdf_docs)} PDF file(s)")

    for glob, label in [("**/*.txt", "TXT"), ("**/*.md", "MD")]:
        loader = DirectoryLoader(docs_dir, glob=glob, loader_cls=TextLoader, show_progress=True)
        try:
            docs = loader.load()
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} {label} file(s)")
        except Exception as e:
            logger.warning(f"{label} loader failed: {e}")

    if not all_docs:
        raise ValueError(f"No documents found in {docs_dir}. Add .pdf/.txt/.md files.")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks


def build_neo4j_vector_store(chunks: list[Document]) -> Neo4jVector:
    embeddings = get_embeddings()
    logger.info("Building Neo4j vector store...")
    vector_store = Neo4jVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        index_name="document_chunks",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
        pre_delete_collection=True,
    )
    logger.info("Neo4j vector store built successfully")
    return vector_store


def build_bm25_index(chunks: list[Document]) -> BM25Okapi:
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"BM25 index saved to {BM25_INDEX_PATH}")
    return bm25


def run_ingestion(docs_dir: str = "data/documents"):
    logger.info("=== Starting ingestion pipeline ===")
    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs)
    build_neo4j_vector_store(chunks)
    build_bm25_index(chunks)
    logger.info(f"=== Ingestion complete — {len(chunks)} chunks stored ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="data/documents")
    args = parser.parse_args()
    run_ingestion(args.docs_dir)