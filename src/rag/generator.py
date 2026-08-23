"""
Prompt Construction and LLM Answer Generation.
This file bridges the gap between retrieved documents and the final user answer.
It contains the prompt templates necessary to instruct the LLM on how to use the provided
context. It handles the injection of context and the user query into the prompt, makes
the API call to the generative model (e.g., GPT-4, Claude), streams the response if necessary,
and enforces constraints (like 'answer only using the provided text').

Classes:
- LLMGenerator: Manages interactions with LLM APIs.

Methods:
- generate_answer(query, retrieved_context): Builds the prompt and calls the LLM.
- generate_stream(query, retrieved_context): Yields the answer token by token for UI streaming.
"""
