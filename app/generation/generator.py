"""
Citation-grounded generator
----------------------------
Formats reranked chunks as numbered context blocks, then calls Claude
with a system prompt that forces [doc_N] citation in every claim.
Parses the response to validate citations exist before returning.
"""
import re

import anthropic
from langchain_core.documents import Document
from loguru import logger

from app.config import settings


SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context documents.

Rules you must follow:
1. Every factual claim must include a citation like [doc_1] or [doc_2, doc_3].
2. Do NOT use any knowledge outside the provided context.
3. If the context does not contain enough information to answer, respond with:
   "The provided documents do not contain sufficient information to answer this question."
4. Be concise and direct. Do not pad your answer.
5. Use the exact document numbers from the context header (e.g. [doc_0], [doc_1]).
"""


def format_context(chunks: list[Document]) -> tuple[str, dict[str, Document]]:
    """
    Format chunks as numbered context blocks.
    Returns the formatted string and a mapping of doc_id → Document.
    """
    lines = []
    doc_map = {}
    for i, chunk in enumerate(chunks):
        doc_id = f"doc_{i}"
        source = chunk.metadata.get("source", "unknown")
        lines.append(f"[{doc_id}] (source: {source})\n{chunk.page_content}")
        doc_map[doc_id] = chunk

    return "\n\n---\n\n".join(lines), doc_map


def extract_citations(answer: str) -> list[str]:
    """Pull all [doc_N] references from the generated answer."""
    return re.findall(r"\[doc_\d+\]", answer)


def generate_answer(
    query: str,
    chunks: list[Document],
) -> dict:
    """
    Call Claude with citation-enforced prompt.

    Returns:
        {
            "answer": str,
            "citations": list[str],
            "sources": list[str],        # unique source file paths
            "context_used": list[str],   # chunk text for each cited doc
        }
    """
    if not chunks:
        return {
            "answer": "No relevant documents were retrieved for your query.",
            "citations": [],
            "sources": [],
            "context_used": [],
        }

    context_text, doc_map = format_context(chunks)

    user_message = f"""Context documents:

{context_text}

---

Question: {query}

Answer (with [doc_N] citations):"""

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    logger.info(f"Calling Claude ({settings.claude_model}) | query='{query[:60]}'")

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = message.content[0].text
    citations = extract_citations(answer)
    unique_citations = list(dict.fromkeys(citations))  # preserve order, dedupe

    sources = list({
        doc_map[c.strip("[]")].metadata.get("source", "unknown")
        for c in unique_citations
        if c.strip("[]") in doc_map
    })

    context_used = [
        doc_map[c.strip("[]")].page_content
        for c in unique_citations
        if c.strip("[]") in doc_map
    ]

    logger.info(
        f"Generated answer | citations={unique_citations} | "
        f"input_tokens={message.usage.input_tokens} "
        f"output_tokens={message.usage.output_tokens}"
    )

    return {
        "answer": answer,
        "citations": unique_citations,
        "sources": sources,
        "context_used": context_used,
    }
