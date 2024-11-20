import os
import json
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

try:
    import constants as const
    from utils import PathHelper, get_logger, timeit
except Exception as e:
    print(e)
    raise RuntimeError("Please run this script from the root directory of the project")

# logger
logger = get_logger(__name__)

# load environment variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# model name for creating embeddings
model_name = const.ENCODING_MODEL_NAME

# function to get text chunks
def get_text_chunks(text):
    """
    Splits text into chunks with specified chunk size and overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# creating embeddings and vectorstore
def get_vectorstore(text_chunks, model_name):
    """
    Creates a vectorstore using the specified embedding model and text chunks.
    """
    logger.info("Creating vectorstore...")
    logger.info(f"Model name: {model_name}")

    # create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # create vectorstore using Chroma
    vectorstore = Chroma.from_texts(
        text_chunks,
        embeddings,
        persist_directory=str(PathHelper.db_dir / const.CHROMA_DB),
    )
    return vectorstore

@timeit
def main():
    """
    Main function to process text files, generate embeddings, and create vectorstore.
    """
    # Load text files
    text_dir = PathHelper.text_dir
    videos = os.listdir(text_dir)
    m_total_docs = len(videos)

    text_chunks = []
    m_success_transcript = 0
    m_failed_docs = 0

    # Process each file
    for v in videos:
        try:
            with open(text_dir / v, encoding="utf8") as f:
                # Load JSON content
                transcript = f.readlines()
                transcript = json.loads(transcript[0])

            # Ensure the transcript is cleaned
            transcript = ''.join(char for char in transcript if char.isprintable())

            if transcript:
                text_chunks_i = get_text_chunks(transcript)
                text_chunks.extend(text_chunks_i)
                m_success_transcript += 1
        except Exception as e:
            logger.error(f"Error processing {v}: {e}")
            m_failed_docs += 1

    logger.info(f"Total docs: {m_total_docs}")
    logger.info(f"Successfully processed transcripts: {m_success_transcript}")
    logger.info(f"Failed docs: {m_failed_docs}")

    # Create vectorstore
    vectorstore = get_vectorstore(text_chunks, model_name=model_name)

    # Create retriever for similarity search
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": const.N_DOCS}
    )
    logger.info(f"Retriever created successfully: {retriever}")

if __name__ == "__main__":
    main()
