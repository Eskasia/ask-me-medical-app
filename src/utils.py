import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# Load environment variables
load_dotenv()

# Dynamically set log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()

# Formatter and level options
FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARN,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}


def get_console_handler():
    """
    Configure console handler for logging.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(log_file: str):
    """
    Configure file handler for logging.
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name, log_file="app.log"):
    """
    Get a configured logger with console and file handlers.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))

    # Avoid adding multiple handlers
    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler(log_file))
    logger.propagate = False
    return logger


class PathHelper:
    """
    Helper class for handling project paths.
    """
    def __init__(self) -> None:
        pass

    root_dir = Path(__file__).parent.parent.absolute()
    src_dir = root_dir / "src"
    data_dir = root_dir / "data"
    entities_dir = data_dir / "entities"
    audio_dir = data_dir / "audio"
    text_dir = data_dir / "text"
    db_dir = root_dir / "db"

    @staticmethod
    def create_directories():
        """
        Create necessary directories, with exception handling.
        """
        try:
            PathHelper.data_dir.mkdir(exist_ok=True, parents=True)
            PathHelper.entities_dir.mkdir(exist_ok=True, parents=True)
            PathHelper.audio_dir.mkdir(exist_ok=True, parents=True)
            PathHelper.text_dir.mkdir(exist_ok=True, parents=True)
            PathHelper.db_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise


# Initialize logger
logger = get_logger(__name__)


def timeit(func):
    """
    Decorator for measuring function execution time.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {total_time:.4f} seconds.")
        return result

    return timeit_wrapper


class MaxPoolingEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    """
    Custom embedding class with average pooling.
    """
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text with average pooling.
        """
        embeddings = self.embed_documents([text])
        return np.mean(embeddings, axis=0).tolist()


def get_connection_string():
    """
    Generate database connection string from environment variables.
    """
    try:
        conn_str = PGVector.connection_string_from_db_params(
            driver=os.getenv("PGVECTOR_DRIVER", "psycopg2"),
            host=os.getenv("PGVECTOR_HOST", "localhost"),
            port=int(os.getenv("PGVECTOR_PORT", "5432")),
            database=os.getenv("PGVECTOR_DATABASE", "postgres"),
            user=os.getenv("PGVECTOR_USER", ""),
            password=os.getenv("PGVECTOR_PASSWORD", ""),
        )
        return conn_str
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        raise ValueError("Missing required environment variables for database connection.")


if __name__ == "__main__":
    # Ensure directories exist
    PathHelper.create_directories()

    # Example usage of logger
    logger.info("Application initialized.")
