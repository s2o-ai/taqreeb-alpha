
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
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(output_dir / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    print("Downloading and extracting audio...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    audio_path = output_dir / "audio.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"File downloaded to {audio_path}")
