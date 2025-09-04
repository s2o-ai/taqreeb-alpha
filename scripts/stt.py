import json
from argparse import ArgumentParser
from pathlib import Path
import torch
import whisper

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("--output-dir", "-o", default="output")
args = parser.parse_args()

file_path: Path = Path(args.file)
output_dir = Path(args.output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Whisper model on device: {device}")
model = whisper.load_model("large-v3-turbo", device=device)
print("Whisper model loaded successfully")

print(f"Starting speech-to-text transcription for: {file_path}")

transcript = model.transcribe(file_path.as_posix(), language="ar")

# Extract segments with timing information
segments: list[dict[str, str | float]] = []
for segment in transcript.get("segments", []):
    segments.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"].strip()
    })

with open(output_dir / "transcript.json", "w") as f:
    json.dump(segments, f)

print(f"Transcription completed. Found {len(segments)} segments")
