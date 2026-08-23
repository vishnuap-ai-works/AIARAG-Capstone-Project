"""
Text Splitting and Chunking Strategies.
This file contains algorithms for breaking down large documents into smaller, semantically
meaningful chunks before embedding. It includes implementations for:
- Recursive Character Text Splitting: Splitting by paragraphs, sentences, and words.
- Semantic Chunking: Grouping text based on embedding similarity to maintain context.
- Token-based Splitting: Ensuring chunks fit strictly within LLM context windows.
Proper chunking is critical for effective retrieval and minimizing noise in the context.

Functions:
- chunk_by_characters(text, chunk_size, overlap): Recursive splitting.
- chunk_by_tokens(text, max_tokens): Token-aware splitting.
- semantic_chunk(text, embedding_model): Groups sentences by semantic similarity.
"""
