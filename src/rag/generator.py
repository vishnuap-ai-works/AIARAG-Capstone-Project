"""
Prompt Construction and LLM Answer Generation.
This file bridges the gap between retrieved documents and the final user answer.
It contains the prompt templates necessary to instruct the LLM on how to use the provided
context.
"""

import math
from config.logging_config import setup_logger
from config.settings import settings
from rag.prompts.prompts import build_generator_prompt

logger = setup_logger(__name__)

class LLMGenerator:
    def __init__(self):
        self.source = getattr(settings, "LLM_SOURCE", "openai").lower()
        
        if self.source == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE_URL)
            self.model = getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini")
        elif self.source == "ollama":
            import ollama
            # Initialize async client (assumes local ollama on default port, or use settings.OLLAMA_BASE_URL)
            base_url = getattr(settings, "OLLAMA_BASE_URL", None)
            if base_url:
                self.client = ollama.AsyncClient(host=base_url)
            else:
                self.client = ollama.AsyncClient()
            self.model = getattr(settings, "OLLAMA_LLM_MODEL", "llama3")
        else:
            raise ValueError(f"Unsupported LLM Source: {self.source}")

    def _build_prompt(self, query: str, context_docs: list[dict]) -> str:
        context_str = "\n\n".join([doc.get("chunk", str(doc)) if isinstance(doc, dict) else str(doc) for doc in context_docs])
        return build_generator_prompt(query, context_str)

    async def generate_text(self, prompt: str) -> str:
        """Generic method to generate text given a prompt (e.g. for query augmentation)."""
        logger.info(f"Generating text using {self.source} ({self.model})")
        try:
            if self.source == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.OPENAI_TEMPERATURE
                )
                return response.choices[0].message.content
            elif self.source == "ollama":
                response = await self.client.generate(
                    model=self.model, 
                    prompt=prompt, 
                    options={"temperature": settings.OLLAMA_TEMPERATURE}
                )
                return response["response"]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    async def generate_answer(
        self, query: str, retrieved_context: list[dict], return_usage: bool = False
    ):
        prompt = self._build_prompt(query, retrieved_context)
        logger.info(f"Generating answer using {self.source} ({self.model})")

        try:
            if self.source == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.OPENAI_TEMPERATURE,
                    logprobs=True
                )
                answer = response.choices[0].message.content
                
                confidence = 1.0
                if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                    content_logprobs = response.choices[0].logprobs.content
                    if content_logprobs:
                        avg_logprob = sum(token.logprob for token in content_logprobs) / len(content_logprobs)
                        confidence = math.exp(avg_logprob)

                if return_usage and hasattr(response, "usage"):
                    return answer, {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "confidence_score": confidence
                    }
                return (
                    answer
                    if not return_usage
                    else (answer, {"prompt_tokens": 0, "completion_tokens": 0, "confidence_score": confidence})
                )
            elif self.source == "ollama":
                # Note: Assuming ollama has a chat method or generate method.
                # using sync ollama wrapper, might block async loop slightly, but OK for MVP.
                response = await self.client.generate(model=self.model, prompt=prompt, options={"temperature": settings.OLLAMA_TEMPERATURE})
                answer = response["response"]
                # Ollama returns eval_count and prompt_eval_count in some versions
                if return_usage:
                    return answer, {
                        "prompt_tokens": response.get("prompt_eval_count", 0),
                        "completion_tokens": response.get("eval_count", 0),
                        "confidence_score": None
                    }
                return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    async def generate_stream(self, query: str, retrieved_context: list[dict]):
        prompt = self._build_prompt(query, retrieved_context)

        if self.source == "openai":
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.OPENAI_TEMPERATURE,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        elif self.source == "ollama":
            stream = await self.client.generate(model=self.model, prompt=prompt, options={"temperature": settings.OLLAMA_TEMPERATURE}, stream=True)
            async for chunk in stream:
                yield chunk["response"]
