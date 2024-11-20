import argparse
import json
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm  # 用於進度條顯示

try:
    import constants as const
    from utils import PathHelper, get_logger
except Exception as e:
    print(e)
    raise RuntimeError("Please run this script from the root directory of the project")

# Logger
logger = get_logger(__name__)


def scroll_to_bottom(driver, timeout=2):
    """
    Scroll to the bottom of the page until no more content loads.
    """
    while True:
        prev_ht = driver.execute_script("return document.documentElement.scrollHeight;")
        driver.execute_script(
            "window.scrollTo(0, document.documentElement.scrollHeight);"
        )
        time.sleep(timeout)
        new_ht = driver.execute_script("return document.documentElement.scrollHeight;")
        if prev_ht == new_ht:
            break


def fetch_video_details(driver, channel_name):
    """
    Fetch video details (URL and title) from a YouTube channel's videos page.
    """
    url = f"https://www.youtube.com/@{channel_name}/videos"
    driver.get(url)

    # Scroll to load all videos
    logger.info(f"Loading videos for channel: {channel_name}")
    scroll_to_bottom(driver)

    videos = []
    try:
        elements = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div#details"))
        )
        for e in elements:
            title = e.find_element(By.CSS_SELECTOR, "a#video-title-link").get_attribute(
                "title"
            )
            vurl = e.find_element(By.CSS_SELECTOR, "a#video-title-link").get_attribute(
                "href"
            )
            videos.append({const.VIDEO_URL: vurl, const.TITLE: title})
    except Exception as e:
        logger.error(f"Error fetching video details: {e}")

    logger.info(f"Found {len(videos)} videos for channel: {channel_name}")
    return videos


def fetch_transcripts(videos, channel_name):
    """
    Fetch transcripts for videos and save them as JSON files.
    """
    for video in tqdm(videos, desc="Fetching transcripts"):
        video_id = video[const.VIDEO_URL].split("v=")[-1]
        video[const.VIDEO_ID] = video_id
        video[const.CHANNEL_NAME] = channel_name

        entity_fname = PathHelper.entities_dir / f"{video[const.VIDEO_ID]}.json"

        # Check if the transcript file already exists
        if entity_fname.exists():
            logger.info(f"Transcript file already exists: {entity_fname}")
            continue

        try:
            # Fetch transcript using YouTubeTranscriptApi
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["zh-TW"]
            )
            video[const.TRANSCRIPT] = transcript
        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {e}")
            video[const.TRANSCRIPT] = []

        # Save video details and transcript
        try:
            with open(entity_fname, "w", encoding="utf8") as f:
                json.dump(video, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error saving transcript file for video {video_id}: {e}")


def main(args):
    channel_name = args.channel_name
    logger.info(f"Processing channel: {channel_name}")

    # Initialize Selenium WebDriver
    driver = webdriver.Chrome()

    try:
        # Fetch video details
        videos = fetch_video_details(driver, channel_name)

        # Fetch transcripts for videos
        fetch_transcripts(videos, channel_name)
    finally:
        driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel-name", type=str, help="Channel Name without @", default="DrTNHuang"
    )
    args = parser.parse_args()
    main(args)
