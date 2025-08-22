import subprocess
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_VIDEO_URL = "https://youtube.com/shorts/vSX3IRxGnNY"
DEFAULT_OUTPUT_FILENAME = "cricket_cover_drive_input.%(ext)s"

def download_video(url: str, output_path: str, quality: str = "best[height<=720]") -> bool:
    try:
        cmd = [
            'yt-dlp',
            '--format', quality,
            '--output', output_path,
            '--no-playlist',
            url
        ]
        
        logger.info(f"Downloading video from: {url}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Quality: {quality}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("Download completed successfully!")
        if result.stdout:
            logger.debug(f"yt-dlp output: {result.stdout}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        if e.stderr:
            logger.error(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return False

def get_video_info(url: str) -> dict:
    """Get video information without downloading"""
    
    try:
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-playlist',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        info = json.loads(result.stdout)
        
        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'width': info.get('width', 0),
            'height': info.get('height', 0),
            'fps': info.get('fps', 0),
            'format': info.get('ext', 'unknown'),
            'filesize': info.get('filesize', 0)
        }
        
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
        return {}

def download_target_video(output_dir: str = ".", quality: str = "best[height<=720]", url=TARGET_VIDEO_URL) -> str:
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up output path
    output_template = str(output_dir / DEFAULT_OUTPUT_FILENAME)
    
    # Get video info first
    logger.info("Getting video information...")
    info = get_video_info(url)
    if info:
        logger.info(f"Video title: {info['title']}")
        logger.info(f"Duration: {info['duration']}s")
        logger.info(f"Resolution: {info['width']}x{info['height']}")
        logger.info(f"FPS: {info['fps']}")
    
    # Download video
    if download_video(url, output_template, quality):
        # Find the actual downloaded file
        for ext in ['.mp4', '.webm', '.mkv', '.avi']:
            potential_file = output_dir / f"cricket_cover_drive_input{ext}"
            if potential_file.exists():
                logger.info(f"Video downloaded successfully: {potential_file}")
                return str(potential_file)
        
        # If no standard extension found, look for any file with the base name
        for file in output_dir.iterdir():
            if file.name.startswith("cricket_cover_drive_input"):
                logger.info(f"Video downloaded successfully: {file}")
                return str(file)
    
    logger.error("Download failed or file not found")
    return None

