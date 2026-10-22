"""
Unit Tests for Evaluation Judge and Ragas Evaluation.
Verifies LLMJudge and RagasJudge functionality, configuration loading,
metric calculation, score sanitization, and report aggregation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest

from config.settings import settings
from eval.golden import aggregate_metrics, run_evaluation
from eval.judge import LLMJudge
from eval.ragas_eval import RagasJudge


def test_llm_judge_initialization():
    """Verify LLMJudge initializes properly with settings."""
    with patch.object(settings, "JUDGE_SOURCE", "ollama"), \
         patch.object(settings, "OLLAMA_JUDGE_MODEL", "qwen2.5:7b"):
        judge = LLMJudge()
        assert judge.source == "ollama"
        assert judge.model == "qwen2.5:7b"


def test_ragas_judge_initialization_ollama():
    """Verify RagasJudge initializes correctly with Ollama source and OLLAMA_JUDGE_MODEL."""
    with patch.object(settings, "JUDGE_SOURCE", "ollama"), \
         patch.object(settings, "OLLAMA_JUDGE_MODEL", "qwen2.5:7b"), \
         patch.object(settings, "OLLAMA_BASE_URL", "http://localhost:11434"):
        judge = RagasJudge()
        assert judge.source == "ollama"
        assert judge.llm is not None
        assert judge.embeddings is not None
        assert len(judge.metrics) == 4


def test_ragas_judge_initialization_openai():
    """Verify RagasJudge initializes correctly with OpenAI source and OPENAI_JUDGE_MODEL."""
    with patch.object(settings, "JUDGE_SOURCE", "openai"), \
         patch.object(settings, "OPENAI_JUDGE_MODEL", "gpt-4o-mini"), \
         patch.object(settings, "OPENAI_API_KEY", "test-key-12345"):
        judge = RagasJudge()
        assert judge.source == "openai"
        assert judge.llm is not None
        assert judge.embeddings is not None


def test_ragas_judge_initialization_huggingface():
    """Verify RagasJudge initializes correctly with HuggingFace source and HUGGINGFACE_JUDGE_MODEL."""
    with patch.object(settings, "JUDGE_SOURCE", "huggingface"), \
         patch.object(settings, "HUGGINGFACE_JUDGE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"), \
         patch.object(settings, "HUGGINGFACE_API_KEY", "hf_test_token"):
        judge = RagasJudge()
        assert judge.source == "huggingface"
        assert judge.llm is not None
        assert judge.embeddings is not None


def test_ragas_score_sanitization():
    """Verify RagasJudge sanitizes None, NaN, inf, strings, and out-of-bound numbers."""
    judge = RagasJudge(source="ollama")
    assert judge._sanitize_score(None) == 0.0
    assert judge._sanitize_score(float("nan")) == 0.0
    assert judge._sanitize_score(float("inf")) == 0.0
    assert judge._sanitize_score("invalid") == 0.0
    assert judge._sanitize_score(0.85) == 0.85
    assert judge._sanitize_score(1.5) == 1.0
    assert judge._sanitize_score(-0.2) == 0.0


def test_ragas_evaluate_batch_mocked():
    """Verify evaluate_batch converts dataset correctly and normalizes scores."""
    judge = RagasJudge(source="ollama")

    fake_eval_df = pd.DataFrame(
        [
            {
                "faithfulness": 0.95,
                "answer_relevancy": 0.88,
                "context_precision": 1.0,
                "context_recall": 0.90,
            }
        ]
    )
    mock_eval_result = MagicMock()
    mock_eval_result.to_pandas.return_value = fake_eval_df

    with patch("ragas.evaluate", return_value=mock_eval_result):
        results = asyncio.run(
            judge.evaluate_batch(
                [
                    {
                        "query": "Where is the company headquartered?",
                        "context": [{"chunk": "The company headquarters is located in San Francisco."}],
                        "generated_answer": "San Francisco.",
                        "expected_answer": "The headquarters is in San Francisco.",
                    }
                ]
            )
        )

        assert len(results) == 1
        res = results[0]
        assert res["ragas_faithfulness"] == 0.95
        assert res["ragas_answer_relevancy"] == 0.88
        assert res["ragas_context_precision"] == 1.0
        assert res["ragas_context_recall"] == 0.90


def test_ragas_evaluate_all_single_item():
    """Verify evaluate_all calls evaluate_batch and returns single dict."""
    judge = RagasJudge(source="ollama")

    fake_scores = [
        {
            "ragas_faithfulness": 0.90,
            "ragas_answer_relevancy": 0.85,
            "ragas_context_precision": 0.95,
            "ragas_context_recall": 0.95,
        }
    ]

    with patch.object(judge, "evaluate_batch", return_value=fake_scores):
        res = asyncio.run(
            judge.evaluate_all(
                query="Test query",
                context=["Test chunk"],
                generated_answer="Test answer",
                expected_answer="Expected test answer",
            )
        )
        assert res["ragas_faithfulness"] == 0.90
        assert res["ragas_answer_relevancy"] == 0.85
        assert res["ragas_context_precision"] == 0.95
        assert res["ragas_context_recall"] == 0.95


def test_aggregate_metrics_with_and_without_ragas():
    """Verify aggregate_metrics handles standard LLM Judge metrics and Ragas additions."""
    standard_results = [
        {
            "latency": 1.0,
            "cost": 0.001,
            "task_success": 1.0,
            "groundedness": 0.9,
            "retrieval_hit": 1.0,
        }
    ]
    metrics1 = aggregate_metrics(standard_results)
    assert metrics1["Total Count"] == 1
    assert metrics1["Task Success Rate"] == "100.00%"
    assert metrics1["Groundedness"] == "90.00%"
    assert "Ragas Faithfulness" not in metrics1

    ragas_enriched_results = [
        {
            "latency": 1.0,
            "cost": 0.001,
            "task_success": 1.0,
            "groundedness": 0.9,
            "retrieval_hit": 1.0,
            "ragas_faithfulness": 0.92,
            "ragas_answer_relevancy": 0.89,
            "ragas_context_precision": 0.95,
            "ragas_context_recall": 0.88,
        }
    ]
    metrics2 = aggregate_metrics(ragas_enriched_results)
    assert metrics2["Ragas Faithfulness"] == "92.00%"
    assert metrics2["Ragas Answer Relevancy"] == "89.00%"
    assert metrics2["Ragas Context Precision"] == "95.00%"
    assert metrics2["Ragas Context Recall"] == "88.00%"
