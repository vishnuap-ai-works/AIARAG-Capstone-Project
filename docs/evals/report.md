# RAG Evaluation Report

## Final Metrics vs Thresholds

| Metric | Acceptable Threshold | Good Threshold | Value | Threshold Evaluation |
|---|---|---|---|---|
| Total Count | N/A | N/A | 50 | N/A |
| Task Success Rate | 0.6 | 0.8 | 96.80% | Good |
| Groundedness | 0.8 | 0.95 | 98.60% | Good |
| Retrieval Hit Rate | 0.7 | 0.9 | 99.60% | Good |
| Cost / Query | 0.01 | 0.005 | $0.00000 | Good |
| Latency (p50) | N/A | N/A | 4.56s | N/A |
| Latency (p95) | 3.0 | 1.5 | 12.22s | Poor |

## Metrics by Path Level

| Path Level | Count | Task Success Rate | Groundedness | Retrieval Hit Rate |
|---|---|---|---|---|
| happy_path | 35 | 98.86% | 99.71% | 100.00% |
| hard_path | 10 | 95.00% | 97.00% | 98.00% |
| edge_path | 5 | 86.00% | 94.00% | 100.00% |

## Threshold Definitions
```json
{
  "task_success_rate": {
    "acceptable": 0.6,
    "good": 0.8
  },
  "groundedness": {
    "acceptable": 0.8,
    "good": 0.95
  },
  "retrieval_rate": {
    "acceptable": 0.7,
    "good": 0.9
  },
  "cost_per_query": {
    "acceptable": 0.01,
    "good": 0.005
  },
  "latency_p95": {
    "acceptable": 3.0,
    "good": 1.5
  }
}
```
