import asyncio
import sys

from config.logging_config import setup_logger
from config.settings import settings
from rag.generator import LLMGenerator
from rag.retriever import DenseRetriever
from rag.vector_store import VectorStoreFactory

logger = setup_logger(__name__)


class InferencePipeline:
    def __init__(self, top_k: int = None):
        logger.info("Initializing InferencePipeline")

        self.top_k = top_k or getattr(settings, "TOP_K", 5)

        reranker = None
        if getattr(settings, "USE_RERANKER", False):
            from rag.reranker import RerankerFactory

            reranker = RerankerFactory.get_reranker()

        self.vector_store = VectorStoreFactory.get_vector_store()
        self.retriever = DenseRetriever(self.vector_store, reranker=reranker)
        self.generator = LLMGenerator()

    async def run(self, query: str):
        logger.info(f"Running inference for query: '{query}'")
        try:
            # 1. Retrieve Context
            logger.info("Step 1: Retrieving context")
            context = await self.retriever.retrieve(query, top_k=self.top_k)

            if not context:
                logger.warning("No context retrieved. Answer may be generic.")

            # 2. Generate Answer
            logger.info("Step 2: Generating answer")
            answer = await self.generator.generate_answer(query, context)

            logger.info("Inference complete.")
            return answer
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            raise


async def main():
    if len(sys.argv) < 2:
        print('Usage: python src/pipeline/inference.py "<your_query_here>"')
        sys.exit(1)

    query = sys.argv[1]

    print("==========================================")
    print("Starting Inference Pipeline...")
    print(f"Query: {query}")
    print("==========================================")

    pipeline = InferencePipeline()
    try:
        answer = await pipeline.run(query)
        print("\n=== Answer ===")
        print(answer)
        print("==============\n")
    except Exception as e:
        print(f"Failed to run inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
