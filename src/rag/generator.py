"""
Prompt Construction and LLM Answer Generation.
This file bridges the gap between retrieved documents and the final user answer.
It contains the prompt templates necessary to instruct the LLM on how to use the provided
context.
"""

from config.logging_config import setup_logger
from config.settings import settings

logger = setup_logger(__name__)


class LLMGenerator:
    def __init__(self):
        self.source = getattr(settings, "LLM_SOURCE", "openai").lower()
        if self.source == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE_URL
            )
            self.model = getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini")
        elif self.source == "ollama":
            import ollama

            self.client = ollama
            self.model = getattr(settings, "OLLAMA_LLM_MODEL", "llama3")
        else:
            raise ValueError(f"Unsupported LLM Source: {self.source}")

    def _build_prompt(self, query: str, retrieved_context: list[dict]) -> str:
        context_str = "\n\n".join([item["chunk"] for item in retrieved_context])
        return f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context below.
If you cannot answer the question based on the context, say "I don't know based on the provided context."

Context:
{context_str}

User Question: {query}
Answer:"""

    async def generate_answer(self, query: str, retrieved_context: list[dict]) -> str:
        prompt = self._build_prompt(query, retrieved_context)
        logger.info(f"Generating answer using {self.source} ({self.model})")

        try:
            if self.source == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content
            elif self.source == "ollama":
                # Note: Assuming ollama has a chat method or generate method.
                # using sync ollama wrapper, might block async loop slightly, but OK for MVP.
                response = self.client.generate(model=self.model, prompt=prompt)
                return response["response"]
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    async def generate_stream(self, query: str, retrieved_context: list[dict]):
        prompt = self._build_prompt(query, retrieved_context)

        if self.source == "openai":
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        elif self.source == "ollama":
            stream = self.client.generate(model=self.model, prompt=prompt, stream=True)
            for chunk in stream:
                yield chunk["response"]
