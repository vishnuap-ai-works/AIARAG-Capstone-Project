"""
Evaluation Judge Logic.
This file implements the `LLMJudge` classes responsible for scoring and grading
generated answers against provided rubrics or expected outcomes.
It includes logic for prompt formatting for the judge LLM, parsing JSON or structured
outputs from the judge, handling retries on malformed outputs, and normalizing scores.

Classes:
- LLMJudge: Uses an LLM to score responses based on criteria like relevance and faithfulness.
- ExactMatchJudge: Simple string comparison for factual questions.

Methods:
- evaluate(query, context, generated_answer, expected_answer): Returns a numerical score and reasoning.
"""
