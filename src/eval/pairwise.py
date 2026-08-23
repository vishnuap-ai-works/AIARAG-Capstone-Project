"""
Pairwise Evaluation Logic.
This module is dedicated to A/B testing two different model responses or two different
RAG pipeline configurations against each other. It handles positional bias swapping.

Functions:
- run_ab_test(model_a_output, model_b_output, query): Uses a judge LLM to pick the better response.
- calculate_win_rate(ab_results): Computes statistical win/loss/tie ratios.
"""
