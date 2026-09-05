import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

from config.logging_config import setup_logger

logger = setup_logger(__name__)


# Template for our file loaders
class BaseDocumentLoader(ABC):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @abstractmethod
    async def load(self) -> str:
        pass


# Reads .txt file format
class TextLoader(BaseDocumentLoader):

    async def load(self) -> str:
        logger.info(f"Loading text file: {self.path.name}")
        if not self.path.exists():
            logger.error(f"File not found: {self.path.name}")
            raise FileNotFoundError(f"File not found: {self.path.name}")

        loop = asyncio.get_event_loop()

        def _read_file():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading text file {self.path.name}: {e}")
                raise

        return await loop.run_in_executor(None, _read_file)


# Reads .md file format
class MarkdownLoader(BaseDocumentLoader):
    """
    Loads markdown files (.md).
    """

    async def load(self) -> str:
        logger.info(f"Loading markdown file: {self.path.name}")
        if not self.path.exists():
            logger.error(f"File not found: {self.path.name}")
            raise FileNotFoundError(f"File not found: {self.path.name}")

        loop = asyncio.get_event_loop()

        def _read_file():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    # Returns raw markdown. You could add logic here to strip markdown syntax if needed.
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading markdown file {self.path.name}: {e}")
                raise

        return await loop.run_in_executor(None, _read_file)


# TODO: Reads .docx file format
# TODO: Reads .doc file format
# TODO: Reads .pdf file format
# TODO: Reads .pdfOcr file format


class DocumentLoadFactory:

    @staticmethod
    def get_loader(file_path: str | Path) -> BaseDocumentLoader:
        path = Path(file_path)
        ext = path.suffix.lower()  # extract extention from a path .txt, .md, .doc

        logger.info(f"Creating loader for file: {path.name} with extension: {ext}")

        if ext == ".txt":
            return TextLoader(path)

        if ext == ".md":
            return MarkdownLoader(path)

        logger.error(f"Unsupported file extension: {ext} for file {path.name}")
        raise ValueError(f"Unsupported file extension: {ext}")
