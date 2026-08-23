"""
Pairwise Evaluation Script (Step 4).
A/B testing tool that runs two different RAG configurations (e.g., BM25 vs Dense retrieval) and uses an LLM judge to determine the winner for a set of queries.

Usage:
`python scripts/run_pairwise.py --model_a gpt-3.5-turbo --model_b gpt-4`
"""
