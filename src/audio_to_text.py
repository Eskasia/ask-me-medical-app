import argparse
import json
import os

from deepmultilingualpunctuation import PunctuationModel
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm

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


def process_audio_to_text(fname_ext, model, punct_model):
    """
    Process a single audio file to text using WhisperModel and restore punctuation.
    """
    fname = fname_ext.split(".")[0]
    audio_path = PathHelper.audio_dir / fname_ext
    try:
        # Load audio
        audio_file = AudioSegment.from_mp3(audio_path)
        logger.info(f"Processing file: {fname_ext}")

        # Transcribe using Whisper
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            initial_prompt="以下是普通話的句子。",
        )
        logger.info(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )

        # Skip non-Chinese/English files
        if info.language not in ("en", "zh"):
            with open(PathHelper.text_dir / f"{fname}.txt", "w") as f:
                json.dump("", f)
            return

        # Process transcript
        seg_i = [[segment.start, segment.end, segment.text] for segment in segments]
        transcript_text = [
            [s[2] for s in seg_i[i : i + 10]] for i in range(0, len(seg_i), 10)
        ]
        transcript_text_restore = [
            punct_model.restore_punctuation(" ".join(text_block))
            for text_block in transcript_text
        ]
        transcript_processed = "".join(transcript_text_restore)

        # Save transcript
        with open(PathHelper.text_dir / f"{fname}.txt", "w", encoding="utf8") as f:
            json.dump(transcript_processed, f)

    except Exception as e:
        logger.error(f"Error processing file {fname_ext}: {e}")


def main(args):
    channel_name = args.channel_name
    logger.info(f"Channel name: {channel_name}")

    # Initialize punctuation model
    punct_model = PunctuationModel()

    # Get list of files
    fnames = [i for i in os.listdir(PathHelper.audio_dir) if i.endswith(".mp3")]
    fnames_has_text = [i.split(".")[0] for i in os.listdir(PathHelper.text_dir) if i.endswith(".txt")]
    fnames_wo_text = [f for f in fnames if f.split(".")[0] not in fnames_has_text]
    logger.info(f"Files with text: {len(fnames_has_text)}")
    logger.info(f"Files without text: {len(fnames_wo_text)}")

    # Filter files based on selected channels
    json_files = os.listdir(PathHelper.entities_dir)
    entities_selected = []
    for jf in json_files:
        try:
            with open(PathHelper.entities_dir / jf, "r") as f:
                ent_i = json.load(f)

            # Filter by channel name
            if channel_name and ent_i.get("channel_name") != channel_name:
                continue
            entities_selected.append(f"{ent_i['video_id']}.mp3")
        except Exception as e:
            logger.error(f"Error loading entity file {jf}: {e}")
            continue

    # Get files to process
    fnames_selected = list(set(fnames_wo_text).intersection(set(entities_selected)))
    logger.info(f"Files selected for processing: {len(fnames_selected)}")

    # Apply limit if specified
    if args.limit > 0:
        fnames_selected = fnames_selected[: args.limit]
    logger.info(f"Files after applying limit: {len(fnames_selected)}")

    # Initialize Whisper model
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Process files
    for fname_ext in tqdm(fnames_selected, desc="Processing audio files"):
        process_audio_to_text(fname_ext, model, punct_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel-name",
        type=str,
        help="Filter by channel name (without @).",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of files to process.",
        default=-1,
    )

    args = parser.parse_args()
    main(args)
