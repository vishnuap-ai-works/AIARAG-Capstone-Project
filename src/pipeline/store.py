"""
SQLite Experiment Tracking.
This module manages the persistence layer for all evaluation runs. It establishes
a connection to a local SQLite database to log inputs, outputs, judge scores,
model configurations, and latency metrics for every single evaluation.

Classes:
- ExperimentTracker: Manages DB connections and table schemas.

Methods:
- log_run(config, metrics): Inserts a new experiment run.
- log_evaluation_case(run_id, query, response, score): Logs individual test case results.
- export_to_csv(): Dumps DB contents for external analysis.
"""
