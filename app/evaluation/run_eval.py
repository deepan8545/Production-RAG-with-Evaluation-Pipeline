import argparse
import asyncio
import json
import sys
from pathlib import Path

# Windows async fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

from app.config import settings
from app.generation.generator import generate_answer
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import rerank


GOLDEN_SET_PATH = Path("eval/golden_set/questions.json")


def load_golden_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Golden test set not found at {path}")
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} golden Q&A pairs from {path}")
    return data


def run_rag_pipeline(query: str, retriever: HybridRetriever) -> dict:
    candidates = retriever.get_relevant_documents(query)
    reranked = rerank(query, candidates)
    result = generate_answer(query, reranked)
    return {
        "answer": result["answer"],
        "contexts": result["context_used"],
    }


def build_ragas_dataset(golden_set: list[dict], retriever: HybridRetriever) -> list[SingleTurnSample]:
    samples = []
    for i, item in enumerate(golden_set):
        logger.info(f"Evaluating {i+1}/{len(golden_set)}: {item['question'][:60]}...")
        try:
            result = run_rag_pipeline(item["question"], retriever)
            contexts = result["contexts"] if result["contexts"] else ["No context retrieved"]
            samples.append(SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=contexts,
                reference=item["ground_truth"],
            ))
        except Exception as e:
            logger.error(f"Pipeline failed for question {i+1}: {e}")
            samples.append(SingleTurnSample(
                user_input=item["question"],
                response="Error",
                retrieved_contexts=["Error"],
                reference=item["ground_truth"],
            ))
    return samples


def check_thresholds(scores: dict) -> tuple[bool, list[str]]:
    thresholds = {
        "faithfulness": settings.faithfulness_threshold,
        "answer_relevancy": settings.answer_relevancy_threshold,
        "context_precision": settings.context_precision_threshold,
    }
    failures = []
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0)
        if score < threshold:
            failures.append(f"{metric}: {score:.3f} < threshold {threshold:.3f}")
    return len(failures) == 0, failures


def run_evaluation(golden_set_path: Path = GOLDEN_SET_PATH):
    logger.info("=== Starting RAGAS evaluation ===")

    try:
        golden_set = load_golden_set(golden_set_path)
    except Exception as e:
        logger.error(f"Failed to load golden set: {e}")
        sys.exit(1)

    try:
        retriever = HybridRetriever()
    except Exception as e:
        logger.error(f"Failed to initialise retriever: {e}")
        sys.exit(1)

    samples = build_ragas_dataset(golden_set, retriever)
    dataset = EvaluationDataset(samples=samples)

    # Claude as judge LLM
    claude_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.0,
        )
    )

    # HuggingFace embeddings — no OpenAI needed
    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # Initialise metrics with our LLM/embeddings
    faithfulness = Faithfulness(llm=claude_llm)
    answer_relevancy = AnswerRelevancy(llm=claude_llm, embeddings=hf_embeddings)
    context_precision = ContextPrecision(llm=claude_llm)

    logger.info("Running RAGAS scoring with Claude as judge...")

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
    except Exception as e:
        logger.error(f"RAGAS evaluate() failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("=== RAGAS Results ===")
    scores = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        try:
            raw = results[metric]
            # RAGAS 0.2.x returns a list of per-sample scores — take the mean
            if isinstance(raw, list):
                score = sum(v for v in raw if v is not None) / max(len([v for v in raw if v is not None]), 1)
            else:
                score = float(raw)
            scores[metric] = score
            logger.info(f"  {metric}: {score:.3f}")
        except Exception as e:
            logger.warning(f"  {metric}: could not read score — {e}")
            scores[metric] = 0.0

    passed, failures = check_thresholds(scores)

    if passed:
        logger.info("=== EVALUATION PASSED — all metrics above threshold ===")
        sys.exit(0)
    else:
        logger.error("=== EVALUATION FAILED ===")
        for f in failures:
            logger.error(f"  FAIL: {f}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    args = parser.parse_args()
    run_evaluation(args.golden_set)