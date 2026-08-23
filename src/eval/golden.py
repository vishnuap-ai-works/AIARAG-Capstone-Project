"""
Golden Set Evaluation Logic.
This script provides utilities for loading, parsing, and running evaluations against
a 'Golden Set' - a curated dataset of inputs and known-good outputs.
It includes functions to iterate over test cases, invoke the generation pipeline,
compare the generated result against the expected answer, and aggregate metrics.

Functions:
- load_golden_set(filepath): Parses the JSONL dataset.
- run_evaluation(golden_dataset, rag_pipeline, judge): Executes the full benchmark run.
- aggregate_metrics(results): Computes average scores, latency, and failure rates.
"""
