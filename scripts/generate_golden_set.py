"""
Golden test set generator
--------------------------
Uses Claude to generate Q&A pairs from your ingested document chunks.
Produces eval/golden_set/questions.json — the input to run_eval.py.

Usage:
    python scripts/generate_golden_set.py
    python scripts/generate_golden_set.py --num-questions 75 --docs-dir data/documents
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import anthropic
from loguru import logger

from app.config import settings


OUTPUT_PATH = Path("eval/golden_set/questions.json")
DOCS_CACHE_PATH = Path("data/docs_cache.pkl")


GENERATION_PROMPT = """You are creating an evaluation dataset for a RAG system.

Given the following document chunk, generate {n} question-answer pairs that:
1. Can be answered ONLY from the given text (not from general knowledge)
2. Are specific and factual, not vague or opinion-based
3. Have clear, verifiable answers present in the chunk

Return ONLY valid JSON in this exact format, nothing else:
[
  {{"question": "...", "ground_truth": "..."}},
  {{"question": "...", "ground_truth": "..."}}
]

Document chunk:
{chunk}
"""


def load_chunks() -> list:
    if not DOCS_CACHE_PATH.exists():
        raise FileNotFoundError(
            "Document cache not found. Run ingestion first:\n"
            "  python -m app.ingestion.ingest"
        )
    with open(DOCS_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def generate_qa_for_chunk(chunk_text: str, client: anthropic.Anthropic, n: int = 2) -> list[dict]:
    """Call Claude to generate Q&A pairs from a single chunk."""
    prompt = GENERATION_PROMPT.format(chunk=chunk_text[:2000], n=n)

    try:
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=800,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if Claude wraps in ```json
        raw = raw.replace("```json", "").replace("```", "").strip()
        pairs = json.loads(raw)
        return pairs if isinstance(pairs, list) else []
    except Exception as e:
        logger.warning(f"Failed to generate QA for chunk: {e}")
        return []


def generate_golden_set(num_questions: int = 50, docs_dir: str = "data/documents"):
    logger.info(f"Generating {num_questions} golden Q&A pairs...")

    chunks = load_chunks()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Sample chunks evenly across the corpus
    sample_size = min(num_questions, len(chunks))
    sampled = random.sample(chunks, sample_size)

    all_qa: list[dict] = []
    for i, chunk in enumerate(sampled):
        logger.info(f"Processing chunk {i+1}/{sample_size}...")
        pairs = generate_qa_for_chunk(chunk.page_content, client, n=1)
        for pair in pairs:
            pair["source"] = chunk.metadata.get("source", "unknown")
        all_qa.extend(pairs)

        if len(all_qa) >= num_questions:
            break

    all_qa = all_qa[:num_questions]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_qa, f, indent=2)

    logger.info(f"Golden set saved: {OUTPUT_PATH} ({len(all_qa)} pairs)")
    logger.info("Sample questions:")
    for qa in all_qa[:3]:
        logger.info(f"  Q: {qa['question']}")
        logger.info(f"  A: {qa['ground_truth'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--docs-dir", type=str, default="data/documents")
    args = parser.parse_args()
    generate_golden_set(args.num_questions, args.docs_dir)
