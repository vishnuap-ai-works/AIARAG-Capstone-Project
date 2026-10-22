# RAG Evaluation Report

## Final Metrics vs Thresholds

| Metric | Acceptable Threshold | Good Threshold | Value | Threshold Evaluation |
|---|---|---|---|---|
| Total Count | N/A | N/A | 2 | N/A |
| Task Success Rate | 0.6 | 0.8 | 100.00% | Good |
| Groundedness | 0.8 | 0.95 | 100.00% | Good |
| Retrieval Hit Rate | 0.7 | 0.9 | 100.00% | Good |
| Cost / Query | 0.01 | 0.005 | $0.00002 | Good |
| Latency (p50) | N/A | N/A | 3.25s | N/A |
| Latency (p95) | 3.0 | 1.5 | 4.92s | Poor |
| Ragas Faithfulness | N/A | N/A | 100.00% | N/A |
| Ragas Answer Relevancy | N/A | N/A | 94.72% | N/A |
| Ragas Context Precision | N/A | N/A | 50.00% | N/A |
| Ragas Context Recall | N/A | N/A | 100.00% | N/A |

## Metrics by Path Level

| Path Level | Count | Task Success Rate | Groundedness | Retrieval Hit Rate |
|---|---|---|---|---|
| happy_path | 2 | 100.00% | 100.00% | 100.00% |

## Ragas Evaluation Metrics

| Ragas Metric | Value |
|---|---|
| Faithfulness | 100.00% |
| Answer Relevancy | 94.72% |
| Context Precision | 50.00% |
| Context Recall | 100.00% |

### Ragas Metrics by Path Level

| Path Level | Count | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|
| happy_path | 2 | 100.00% | 94.72% | 50.00% | 100.00% |

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

## Cost Evaluation

| Total Questions | Topk Value | Total Input Tokens | System Prompt Tokens | Retrieved Chunks Tokens | Question Embeding Tokens | Answer Generated Tokens | Cost | Avg Latency |
|---|---|---|---|---|---|---|---|---|
| 2 | 5 | 3104.0 | 424.0 | 2654.0 | 26.0 | 32.0 | $0.000048 | 3.25s |
