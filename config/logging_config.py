"""
Standardized Logging Configuration.
Sets up JSON or standard formatting for logs across the entire RAG pipeline.
Ensures that errors, warnings, and evaluation metrics are properly piped to standard out
or log files for production monitoring.

Functions:
- setup_logger(name): Returns a pre-configured Python logger instance.
"""

import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with a standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
