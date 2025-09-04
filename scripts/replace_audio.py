from argparse import ArgumentParser
from pathlib import Path
import subprocess


parser = ArgumentParser()
parser.add_argument("video_path")
parser.add_argument("audio_path")
parser.add_argument("--output-dir", "-o", default="output")
args = parser.parse_args()

video_path = Path(args.video_path)
audio_path = Path(args.audio_path)
output_dir = Path(args.output_dir)

output_path = output_dir / "final_video.mp4"

cmd = [
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    "-i", str(video_path),
    "-i", str(audio_path),
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    str(output_path)
]

process = subprocess.run(cmd, capture_output=True, text=True)
if process.returncode != 0:
    raise RuntimeError(f"FFmpeg failed: {process.stderr}")

print(f"Video with replaced audio created: {output_path}")
