import argparse
import os
import json

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.pgvector import PGVector
from tqdm import tqdm

try:
    import constants as const
    from utils import PathHelper, get_connection_string, get_logger, timeit
except Exception as e:
    print(e)
    raise RuntimeError("Please run this script from the root directory of the project")

# Logger
logger = get_logger(__name__)

# Load environment variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Model name for creating embeddings
model_name = const.ENCODING_MODEL_NAME


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """
    Calculate the number of tokens in a given string using the specified encoding.
    """
    if not string:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def process_videos_to_dataframe(text_dir, text_splitter) -> pd.DataFrame:
    """
    Process text files into a DataFrame with video_id and content.
    """
    videos = os.listdir(text_dir)
    video_ids = [v.split(".")[0] for v in videos]
    data = []

    for video_id in tqdm(video_ids, desc="Processing videos"):
        try:
            with open(text_dir / f"{video_id}.txt", encoding="utf8") as f:
                transcript = f.readlines()
                text = json.loads(transcript[0])  # Use json.loads instead of eval

            token_len = num_tokens_from_string(text)
            if token_len <= 512:
                data.append([video_id, text])
            else:
                # Split text into chunks
                split_texts = text_splitter.split_text(text)
                for chunk in split_texts:
                    data.append([video_id, chunk])
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")

    return pd.DataFrame(data, columns=["video_id", "content"])


@timeit
def main(args):
    # Initialize embeddings and text splitter
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100, length_function=len
    )

    # Load text data into DataFrame
    df_videos = process_videos_to_dataframe(PathHelper.text_dir, text_splitter)

    # Load data as LangChain documents
    loader = DataFrameLoader(df_videos, page_content_column="content")
    docs = loader.load()

    # Initialize database
    if args.db == "pgvector":
        db = PGVector(
            collection_name=const.COLLECTION_NAME,
            connection_string=get_connection_string(),
            embedding_function=embeddings,
        )

        # Add documents in batches
        batch_size = 100
        for i in tqdm(range(0, len(docs), batch_size), desc="Loading to PGVector"):
            db.add_documents(docs[i : i + batch_size])
        logger.info("Documents successfully loaded into PGVector.")

    elif args.db == "chroma":
        page_contents = [doc.page_content for doc in docs]
        db = Chroma.from_texts(
            page_contents,
            embeddings,
            persist_directory=str(PathHelper.db_dir / const.CHROMA_DB),
        )
        logger.info("Documents successfully loaded into Chroma.")

    else:
        raise ValueError(f"Unsupported database type: {args.db}")

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=str,
        choices=["pgvector", "chroma"],
        default="pgvector",
        help="Specify the database type to use ('pgvector' or 'chroma').",
    )

    args = parser.parse_args()
    main(args)
