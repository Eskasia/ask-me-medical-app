import json
import os

from moviepy.editor import AudioFileClip
from pytube import YouTube
from tqdm import tqdm

try:
    import constants as const
    from utils import PathHelper, get_logger
except Exception as e:
    print(e)
    raise RuntimeError("Please run this script from the root directory of the project")

logger = get_logger(__name__)


def download_and_convert_to_mp3(video_url, output_path):
    """
    Download a YouTube video and convert it to an MP3 file.
    """
    # Get video ID from URL
    video_id = video_url.split("=")[-1]
    output_mp3 = f"{output_path}/{video_id}.mp3"
    temp_video = f"{output_path}/temp.mp4"

    # Skip if MP3 already exists
    if os.path.exists(output_mp3):
        logger.info(f"MP3 already exists for video ID: {video_id}")
        return

    try:
        # Download video
        yt = YouTube(video_url)
        video = yt.streams.get_highest_resolution()
        video.download(output_path, filename="temp.mp4")
        logger.info(f"Downloaded video: {video_id}")

        # Convert to MP3
        video_clip = AudioFileClip(temp_video)
        video_clip.write_audiofile(output_mp3)
        video_clip.close()
        logger.info(f"Converted to MP3: {video_id}")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
    finally:
        # Clean up temporary video file
        if os.path.exists(temp_video):
            os.remove(temp_video)
            logger.info(f"Temporary file removed: {temp_video}")


def main():
    # List all JSON files in entities directory
    entities = [i for i in os.listdir(PathHelper.entities_dir) if i.endswith(".json")]

    m_docs = len(entities)
    m_docs_wo_transcript = 0
    m_docs_failed = 0

    for entity in tqdm(entities, desc="Processing entities"):
        try:
            with open(PathHelper.entities_dir / entity, "r", encoding="utf8") as f:
                data = json.load(f)

            # Download and convert if no transcript
            if not data.get(const.TRANSCRIPT):  # Fix incorrect spelling of "TRANSCRIPT"
                m_docs_wo_transcript += 1
                download_and_convert_to_mp3(data[const.VIDEO_URL], PathHelper.audio_dir)

        except Exception as e:
            logger.error(f"Error processing entity {entity}: {e}")
            m_docs_failed += 1

    # Log summary
    logger.info(f"Total documents: {m_docs}")
    logger.info(f"Documents without transcript: {m_docs_wo_transcript}")
    logger.info(f"Documents failed: {m_docs_failed}")


if __name__ == "__main__":
    main()
