"""
End-to-End RAG Pipeline Orchestrator.
Runs the entire RAG system against a golden set to measure overall system accuracy.
Tests the synergy of chunking, retrieval, and generation.

Usage:
`python scripts/run_rag_eval.py`
"""

import asyncio
import json
import os
import sys

# Ensure src/ and root are in PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from config.logging_config import setup_logger
from eval.golden import aggregate_metrics, load_golden_set, run_evaluation
from eval.judge import LLMJudge
from pipeline.inference import InferencePipeline

logger = setup_logger(__name__)


def generate_report(
    metrics: dict, thresholds: dict, report_path: str, results: list = None
):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def evaluate_metric(
        metric_val: float, thresh_dict: dict, reverse: bool = False
    ) -> str:
        # if reverse, lower is better (e.g. latency, cost)
        acceptable = thresh_dict.get("acceptable", 0)
        good = thresh_dict.get("good", 0)

        if reverse:
            if metric_val <= good:
                return "Good"
            elif metric_val <= acceptable:
                return "Acceptable"
            else:
                return "Poor"
        else:
            if metric_val >= good:
                return "Good"
            elif metric_val >= acceptable:
                return "Acceptable"
            else:
                return "Poor"

    # Extract numerical values from metrics for comparison
    tsr = float(metrics.get("Task Success Rate", "0").strip("%")) / 100.0
    grd = float(metrics.get("Groundedness", "0").strip("%")) / 100.0
    rhr = float(metrics.get("Retrieval Hit Rate", "0").strip("%")) / 100.0
    cost = float(metrics.get("Cost / Query", "0").strip("$"))
    latency = float(metrics.get("Latency (p95)", "0").strip("s"))

    evals = {
        "Task Success Rate": evaluate_metric(
            tsr, thresholds.get("task_success_rate", {}), False
        ),
        "Groundedness": evaluate_metric(grd, thresholds.get("groundedness", {}), False),
        "Retrieval Hit Rate": evaluate_metric(
            rhr, thresholds.get("retrieval_rate", {}), False
        ),
        "Cost / Query": evaluate_metric(
            cost, thresholds.get("cost_per_query", {}), True
        ),
        "Latency (p95)": evaluate_metric(
            latency, thresholds.get("latency_p95", {}), True
        ),
    }

    with open(report_path, "w") as f:
        f.write("# RAG Evaluation Report\n\n")
        f.write("## Final Metrics vs Thresholds\n\n")
        f.write(
            "| Metric | Acceptable Threshold | Good Threshold | Value | Threshold Evaluation |\n"
        )
        f.write("|---|---|---|---|---|\n")

        # Helper to get threshold string safely
        def get_thresh_str(metric_key, thresh_type):
            mapping = {
                "Task Success Rate": "task_success_rate",
                "Groundedness": "groundedness",
                "Retrieval Hit Rate": "retrieval_rate",
                "Cost / Query": "cost_per_query",
                "Latency (p95)": "latency_p95",
            }
            if metric_key in mapping:
                t = thresholds.get(mapping[metric_key], {})
                val = t.get(thresh_type, "N/A")
                return str(val)
            return "N/A"

        for key, val in metrics.items():
            eval_status = evals.get(key, "N/A")
            acc_str = get_thresh_str(key, "acceptable")
            good_str = get_thresh_str(key, "good")
            f.write(f"| {key} | {acc_str} | {good_str} | {val} | {eval_status} |\n")

        if results:
            f.write("\n## Metrics by Path Level\n\n")
            f.write(
                "| Path Level | Count | Task Success Rate | Groundedness | Retrieval Hit Rate |\n"
            )
            f.write("|---|---|---|---|---|\n")

            levels = {}
            for r in results:
                lvl = r.get("level", "unknown")
                if lvl not in levels:
                    levels[lvl] = {"count": 0, "ts": 0.0, "gr": 0.0, "rh": 0.0}
                levels[lvl]["count"] += 1
                levels[lvl]["ts"] += r.get("task_success", 0.0)
                levels[lvl]["gr"] += r.get("groundedness", 0.0)
                levels[lvl]["rh"] += r.get("retrieval_hit", 0.0)

            for lvl, data in levels.items():
                c = data["count"]
                ts_pct = (data["ts"] / c) if c > 0 else 0
                gr_pct = (data["gr"] / c) if c > 0 else 0
                rh_pct = (data["rh"] / c) if c > 0 else 0
                f.write(
                    f"| {lvl} | {c} | {ts_pct:.2%} | {gr_pct:.2%} | {rh_pct:.2%} |\n"
                )

        # Add Ragas Evaluation Metrics if available
        if results and "ragas_faithfulness" in results[0]:
            f.write("\n## Ragas Evaluation Metrics\n\n")
            f.write("| Ragas Metric | Value |\n")
            f.write("|---|---|\n")
            f.write(f"| Faithfulness | {metrics.get('Ragas Faithfulness', 'N/A')} |\n")
            f.write(f"| Answer Relevancy | {metrics.get('Ragas Answer Relevancy', 'N/A')} |\n")
            f.write(f"| Context Precision | {metrics.get('Ragas Context Precision', 'N/A')} |\n")
            f.write(f"| Context Recall | {metrics.get('Ragas Context Recall', 'N/A')} |\n")

            f.write("\n### Ragas Metrics by Path Level\n\n")
            f.write(
                "| Path Level | Count | Faithfulness | Answer Relevancy | Context Precision | Context Recall |\n"
            )
            f.write("|---|---|---|---|---|---|\n")

            ragas_levels = {}
            for r in results:
                lvl = r.get("level", "unknown")
                if lvl not in ragas_levels:
                    ragas_levels[lvl] = {
                        "count": 0,
                        "f": 0.0,
                        "ar": 0.0,
                        "cp": 0.0,
                        "cr": 0.0,
                    }
                ragas_levels[lvl]["count"] += 1
                ragas_levels[lvl]["f"] += r.get("ragas_faithfulness", 0.0)
                ragas_levels[lvl]["ar"] += r.get("ragas_answer_relevancy", 0.0)
                ragas_levels[lvl]["cp"] += r.get("ragas_context_precision", 0.0)
                ragas_levels[lvl]["cr"] += r.get("ragas_context_recall", 0.0)

            for lvl, data in ragas_levels.items():
                c = data["count"]
                f_pct = (data["f"] / c) if c > 0 else 0
                ar_pct = (data["ar"] / c) if c > 0 else 0
                cp_pct = (data["cp"] / c) if c > 0 else 0
                cr_pct = (data["cr"] / c) if c > 0 else 0
                f.write(
                    f"| {lvl} | {c} | {f_pct:.2%} | {ar_pct:.2%} | {cp_pct:.2%} | {cr_pct:.2%} |\n"
                )

        f.write("\n## Threshold Definitions\n")
        f.write("```json\n")
        f.write(json.dumps(thresholds, indent=2))
        f.write("\n```\n")

        # Add Cost Evaluation Table
        if results:
            from config.settings import settings

            total_q = len(results)
            top_k = getattr(settings, "TOP_K", "N/A")
            tot_in_tokens = sum(r.get("total_input_tokens", 0) for r in results)
            sys_tokens = sum(r.get("tokens_system_prompt", 0) for r in results)
            ret_tokens = sum(r.get("tokens_on_retrieved_chunks", 0) for r in results)
            emb_tokens = sum(r.get("embedding_token_consumption", 0) for r in results)
            ans_tokens = sum(r.get("tokens_on_completion", 0) for r in results)
            cost = sum(r.get("cost", 0) for r in results)
            avg_latency = (
                sum(r.get("total_time_taken", 0) for r in results) / total_q
                if total_q > 0
                else 0
            )

            f.write("\n## Cost Evaluation\n\n")
            f.write(
                "| Total Questions | Topk Value | Total Input Tokens | System Prompt Tokens | Retrieved Chunks Tokens | Question Embeding Tokens | Answer Generated Tokens | Cost | Avg Latency |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            f.write(
                f"| {total_q} | {top_k} | {tot_in_tokens:.1f} | {sys_tokens:.1f} | {ret_tokens:.1f} | {emb_tokens:.1f} | {ans_tokens:.1f} | ${cost:.6f} | {avg_latency:.2f}s |\n"
            )

    print(f"\nReport successfully generated at: {report_path}")


async def main():
    print("==========================================")
    print("Starting Comprehensive RAG Evaluation...")
    print("==========================================\n")

    golden_path = os.path.join(root_dir, "data", "golden_set.jsonl")
    thresholds_path = os.path.join(root_dir, "data", "evals", "thresholds.json")
    report_path = os.path.join(root_dir, "docs", "evals", "report.md")

    # Load dataset
    golden_dataset = load_golden_set(golden_path)
    if not golden_dataset:
        print("Error: Could not load golden dataset.")
        sys.exit(1)

    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid limit '{sys.argv[1]}'. Running full evaluation.")

    if limit and limit > 0:
        golden_dataset = golden_dataset[:limit]
        print(
            f"Successfully loaded {len(golden_dataset)} items (limited from {golden_path})\n"
        )
    else:
        print(f"Successfully loaded {len(golden_dataset)} items from {golden_path}\n")

    # Load thresholds
    try:
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load thresholds: {e}")
        thresholds = {}

    from config.settings import settings

    # Initialize components
    pipeline = InferencePipeline()
    judge = LLMJudge()
    ragas_judge = None
    if getattr(settings, "ENABLE_RAGAS_EVAL", False):
        try:
            from eval.ragas_eval import RagasJudge

            ragas_judge = RagasJudge()
            print(f"Ragas evaluation enabled (Judge source: {ragas_judge.source})\n")
        except Exception as e:
            print(f"Warning: Could not initialize RagasJudge: {e}\n")
            ragas_judge = None

    # Run evaluation
    results = await run_evaluation(golden_dataset, pipeline, judge, ragas_judge=ragas_judge)

    # Save all results into a single json file in data/evals
    evals_dir = os.path.join(root_dir, "data", "evals")
    os.makedirs(evals_dir, exist_ok=True)
    file_path = os.path.join(evals_dir, "all_results.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    # Aggregate and print metrics
    metrics = aggregate_metrics(results)

    print("\n==========================================")
    print("Evaluation Complete. Final Metrics:")
    print("==========================================")
    for key, value in metrics.items():
        print(f"{key.ljust(25)}: {value}")
    print("==========================================")

    # Generate report
    generate_report(metrics, thresholds, report_path, results)


if __name__ == "__main__":
    asyncio.run(main())
