"""
Standard Evaluation Runner (Step 3).
Main entry point for running standard, single-metric evaluations. It loads a dataset,
instantiates the specified evaluation judge, processes outputs, and logs final scores.

Usage:
`python scripts/run_eval.py --dataset data/golden_set.jsonl --judge llm_judge`
"""
