import argparse
import json
import os

from deepmultilingualpunctuation import PunctuationModel
from dotenv import load_dotenv
from tqdm import tqdm  # 用於進度條顯示

try:
    from utils import PathHelper, get_logger
except Exception as e:
    print(e)
    raise RuntimeError("Please run this script from the root directory of the project")

# Load environment variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Logger
logger = get_logger(__name__)

# Initialize Punctuation Model
logger.info("Initializing punctuation model...")
punct_model = PunctuationModel()


# Process functions
def preprocess_transcript(transcript):
    """
    Add punctuation to transcript and merge into one document.
    """
    n = 5  # Number of lines to process together
    transcript_text = [transcript[i : i + n] for i in range(0, len(transcript), n)]

    # Restore punctuation
    transcript_text_restore = [
        punct_model.restore_punctuation(" ".join(text_block))
        for text_block in transcript_text
    ]

    # Merge into a single document
    return "\n".join(transcript_text_restore)


def main(args):
    # List all JSON files in entities directory
    entities = [i for i in os.listdir(PathHelper.entities_dir) if i.endswith(".json")]
    logger.info(f"Total JSON files: {len(entities)}")

    # Track processed transcripts
    transcripts_processed = 0

    # Process each file with a progress bar
    for jf in tqdm(entities, desc="Processing transcripts"):
        fname = jf.split(".")[0]
        try:
            # Skip if output file already exists
            output_path = PathHelper.text_dir / f"{fname}.txt"
            if output_path.exists():
                logger.info(f"File already exists, skipping: {fname}")
                continue

            # Load JSON content
            with open(PathHelper.entities_dir / jf, "r", encoding="utf8") as f:
                ent_i = json.load(f)

            # Check for transcript in JSON
            if ent_i.get("transcript"):
                transcript_text = [t["text"] for t in ent_i["transcript"]]

                # Preprocess the transcript
                transcript = preprocess_transcript(transcript_text)

                # Save the processed transcript
                with open(output_path, "w", encoding="utf8") as f:
                    json.dump(transcript, f)

                transcripts_processed += 1

        except Exception as e:
            logger.error(f"Error processing file {jf}: {e}")
            continue

    # Log summary
    logger.info(f"Extracted and processed transcripts from {transcripts_processed} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the number of files to process. Default: -1 (no limit).",
    )
    args = parser.parse_args()
    main(args)
