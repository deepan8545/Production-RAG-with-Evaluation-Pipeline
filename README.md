# Production RAG — Hybrid Retrieval + Evaluation Pipeline

Production-grade RAG system with:
- Hybrid BM25 + Neo4j vector retrieval with RRF fusion
- Cross-encoder reranking (ms-marco-MiniLM)
- Citation-grounded generation via Claude (Anthropic)
- CI-gated RAGAS evaluation pipeline (faithfulness, answer relevancy, context precision)
- FastAPI REST layer

## Stack

| Layer | Tool |
|---|---|
| Vector store | Neo4j (local Desktop or AuraDB) |
| BM25 | rank-bm25 |
| Reranker | sentence-transformers cross-encoder |
| LLM | Claude 3.5 Sonnet (Anthropic) |
| Embeddings | OpenAI text-embedding-3-small or HuggingFace |
| Evaluation | RAGAS |
| API | FastAPI |

---

## Setup (Windows — Git Bash)

### 1. Prerequisites

Install these first:
- Python 3.11: https://python.org/downloads (check "Add to PATH")
- Git: https://git-scm.com
- Neo4j Desktop: https://neo4j.com/download
- VS Code: https://code.visualstudio.com

### 2. Neo4j Setup

1. Open Neo4j Desktop → New Project → Add → Local DBMS
2. Set password to `password123`
3. Click Start
4. Confirm it's running at bolt://localhost:7687

### 3. Project Setup

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/production-rag.git
cd production-rag

# Create virtual environment
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows

# Install dependencies
pip install -r requirements.txt

# Copy env and fill in your keys
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 4. Add Documents

Drop `.pdf`, `.txt`, or `.md` files into `data/documents/`.
The sample file `rag_overview.md` is already there to test with.

### 5. Run Ingestion

```bash
python -m app.ingestion.ingest --docs-dir data/documents
```

This:
- Loads and chunks all documents
- Embeds chunks and stores in Neo4j
- Builds and saves BM25 index to `data/bm25_index.pkl`

### 6. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the Swagger UI.

### 7. Test a Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Reciprocal Rank Fusion?"}'
```

### 8. Generate Golden Test Set

```bash
python scripts/generate_golden_set.py --num-questions 50
```

### 9. Run RAGAS Evaluation

```bash
python -m app.evaluation.run_eval
```

Exits 0 (pass) or 1 (fail) based on thresholds in `.env`.

---

## GitHub Actions CI

The `.github/workflows/rag_eval.yml` workflow:
1. Spins up Neo4j as a service container
2. Ingests documents
3. Runs RAGAS evaluation
4. Fails the PR if any metric is below threshold

Add these secrets in your GitHub repo settings:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

---

## Free embeddings (no OpenAI key needed)

In `.env`, set:
```
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

And remove `OPENAI_API_KEY`. The model downloads on first run (~90MB).

---

## Evaluation Thresholds (`.env`)

| Metric | Default threshold |
|---|---|
| faithfulness | 0.85 |
| answer_relevancy | 0.80 |
| context_precision | 0.75 |

---

## Project Structure

```
production-rag/
├── app/
│   ├── config.py                  # pydantic-settings config
│   ├── main.py                    # FastAPI app
│   ├── ingestion/
│   │   ├── ingest.py              # document loading, chunking, indexing
│   │   └── embeddings.py          # embeddings factory (OpenAI / HF)
│   ├── retrieval/
│   │   ├── hybrid_retriever.py    # BM25 + Neo4j vector + RRF fusion
│   │   └── reranker.py            # cross-encoder reranking
│   ├── generation/
│   │   └── generator.py           # citation-grounded Claude generation
│   └── evaluation/
│       └── run_eval.py            # RAGAS scoring + CI gate
├── data/
│   └── documents/                 # drop your corpus here
├── eval/
│   └── golden_set/
│       └── questions.json         # golden Q&A pairs
├── scripts/
│   └── generate_golden_set.py     # generate golden set via Claude
├── .github/
│   └── workflows/
│       └── rag_eval.yml           # CI evaluation gate
├── requirements.txt
├── .env.example
└── README.md
```
