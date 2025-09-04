
from argparse import ArgumentParser
from pathlib import Path
import yt_dlp

parser = ArgumentParser()
parser.add_argument("video_url")
parser.add_argument("--output-dir", "-o", default="output")
args = parser.parse_args()

video_url: str = args.video_url
output_dir = Path(args.output_dir)

ydl_opts = {
    "format": "bv*[ext=mp4]/best[ext=mp4]",
    "outtmpl": str(output_dir / "video.%(ext)s"),
    "quiet": True,
    "no_warnings": True,
}

print("Downloading video...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

video_path = output_dir / "video.mp4"

if not video_path.exists():
    raise FileNotFoundError(f"Video file not found: {video_path}")

print(f"File downloaded to {video_path}")
