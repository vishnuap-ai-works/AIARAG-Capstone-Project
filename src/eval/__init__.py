"""
Evaluation Module Initialization.
This module contains the core components for evaluating LLM-generated responses.
It exposes standard interfaces for judging response quality, managing golden datasets,
running pairwise comparisons between models, and creating AI critics.
"""

from eval.golden import aggregate_metrics, load_golden_set, run_evaluation
from eval.judge import LLMJudge
from eval.ragas_eval import RagasJudge

__all__ = [
    "LLMJudge",
    "RagasJudge",
    "load_golden_set",
    "run_evaluation",
    "aggregate_metrics",
]

