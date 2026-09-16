import logging

logger = logging.getLogger(__name__)

def build_generator_prompt(query: str, context_str: str) -> str:
    try:
        logger.info(f"Building generator prompt for query: {query}")
        return f"""You are a friendly, conversational AI assistant. Use the following context to answer the user's question naturally and clearly. Talk to the user like a human expert.
If the answer is not in the context, politely say "I don't know based on the provided context.if you want to ask any questions to the user feel free to ask for understanding better context"

Context:
{context_str}

User Question: {query}
Answer:"""
    except Exception as e:
        logger.error(f"Error building generator prompt: {str(e)}")
        raise

def build_comprehensive_eval_prompt(query: str, expected_answer: str, context_str: str, generated_answer: str) -> str:
    try:
        logger.info(f"Building comprehensive eval prompt for query: {query}")
        return f"""
You are an expert evaluator. Evaluate the generated answer across three metrics: Task Success, Groundedness, and Retrieval Hit.
Evaluate based on SEMANTIC MEANING and FACTUAL ACCURACY, not exact wording.

1. Task Success: Does the generated answer correctly and fully address the user's query based on the expected answer? 
   Score 0 to 1 (e.g. 1.0 for completely correct even if phrased differently).
2. Groundedness: Is the generated answer fully supported by the provided context? 
   Score 0 to 1 (e.g. 1.0 if fully supported semantically).
3. Retrieval Hit: Does the provided context contain sufficient information to deduce the expected answer? 
   Score 0 to 1 (e.g. 1.0 if all necessary facts are present).

Respond ONLY with a JSON object containing the fields "task_success", "groundedness", and "retrieval_hit" with their respective float scores.

Query: {query}
Expected Answer: {expected_answer}
Context: {context_str}
Generated Answer: {generated_answer}
"""
    except Exception as e:
        logger.error(f"Error building comprehensive eval prompt: {str(e)}")
        raise

def build_multi_query_prompt(query: str, num_queries: int = 3) -> str:
    try:
        logger.info(f"Building multi-query prompt for query: {query}")
        return f"""You are an AI language model assistant. Your task is to generate {num_queries} 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. DO NOT number them. DO NOT add any extra text.

Original question: {query}"""
    except Exception as e:
        logger.error(f"Error building multi-query prompt: {str(e)}")
        raise

def build_query_decomposition_prompt(query: str) -> str:
    try:
        logger.info(f"Building query decomposition prompt for query: {query}")
        return f"""You are a helpful assistant that generates multiple sub-questions related to an input question. 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. 
Generate multiple search queries related to: {query} \n
Output (3 queries): Provide these alternative questions separated by newlines. DO NOT number them. DO NOT add any extra text."""
    except Exception as e:
        logger.error(f"Error building query decomposition prompt: {str(e)}")
        raise
