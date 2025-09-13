from dataclasses import dataclass
import io
import os
import json
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
from typing import Callable, Self
from urllib.parse import urlparse, parse_qs

from kokoro import KPipeline
from pydub import AudioSegment
import torch
import numpy as np
import whisper
import yt_dlp
from transformers import MarianMTModel, MarianTokenizer

REVISING_TAG = "--- REVISING ---\n"

@dataclass
class Target:
    out: Path
    deps: list[Self]
    generator: Callable[[Path], None]

    def is_ready(self) -> bool:
        for dep in self.deps:
            if not dep.out.exists():
                return False
            if self.needs_revision():
              with open(dep.out, "r") as f:
                  if f.readline().startswith(REVISING_TAG):
                      return False

        return True

    def is_done(self) -> bool:
        return self.out.exists()

    def gen(self) -> None:
        self.generator(self.out)

    def needs_revision(self) -> bool:
        return self.out.suffix in ["json"]


def get_audio(video_url: str, out: Path) -> None:
    if out.exists():
        print(f"Audio file is already downloaded at {out}")

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out.parent / "".join(out.name.split(".")[:-1])),
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

    if not out.exists():
        raise FileNotFoundError(f"Audio file not found: {out}")

    print(f"File downloaded to {out}")


def get_video(video_url: str, out: Path) -> None:
    if out.exists():
        print(f"Video is already downloaded at {out}")

    ydl_opts = {
        "format": "bv*[ext=mp4]/best[ext=mp4]",
        "outtmpl": str(out),
        "quiet": True,
        "no_warnings": True,
    }

    print("Downloading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    if not out.exists():
        raise FileNotFoundError(f"Video file not found: {out}")

    print(f"File downloaded to {out}")


def convert_stt(audio: Path, out: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper model on device: {device}")
    model = whisper.load_model("large-v3-turbo", device=device)
    print("Whisper model loaded successfully")

    print(f"Starting speech-to-text transcription for: {audio}")

    transcript = model.transcribe(audio.as_posix(), language="ar")

    # Extract segments with timing information
    segments: list[dict[str, str | float]] = []
    for segment in transcript.get("segments", []):
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })

    with open(out, "w", encoding="utf8") as f:
        transcript = REVISING_TAG
        transcript += json.dumps(segments, ensure_ascii=False, indent=2)
        f.write(transcript)

    print(f"Transcription completed. Found {len(segments)} segments")


def translate_ar2en(transcript: Path, out: Path) -> None:
    # MODEL_ID = "Helsinki-NLP/opus-mt-ar-en"
    MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-ar-en"

    print(f"Loading translation model: {MODEL_ID}")

    device = "cuda" if torch.cuda.is_available() else "auto"
    tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
    model = MarianMTModel.from_pretrained(MODEL_ID, device_map=device)

    print("Translation model loaded successfully")


    with open(transcript, "r") as f:
        segments: list[dict[str, str | float]] = json.load(f)

    print(f"Starting Arabic to English translation for {len(segments)} segments")

    # Create a copy of the transcript for translation
    batch_size = 256
    texts = [segment["text"] for segment in segments]
    translated_texts = []
    for i in range(0, len(texts), batch_size):
      print("Tokenizing and translating texts...")
      batch = texts[i : min(len(texts), i + batch_size)]
      inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
      translated_tokens = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id)
      translated_texts += [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

    # Reconstruct segments with translations
    text_idx = 0
    for segment in segments:
        if not segment["text"].strip(): # type: ignore
            continue
        translation = translated_texts[text_idx] if text_idx < len(translated_texts) else segment["test"]
        segment["text"] = translation
        text_idx += 1

    print("Writing translated segments")
    with open(out, "w", encoding="utf8") as f:
        translated = REVISING_TAG
        translated += json.dumps(segments, ensure_ascii=False, indent=2)
        f.write(translated)

    print("Translation completed successfully")



def convert_tts(transcript: Path, out_audio: Path, out_meta: Path, voice: str) -> None:
    print("Initializing TTS model")

    # _ = "gpu" if torch.cuda.is_available() else "cpu" # TODO: run TTS in GPU
    tts = KPipeline(lang_code="a")

    print("TTS model loaded successfully")

    with open(transcript, "r", encoding="utf8") as f:
        segments: list[dict[str, str | float]] = json.load(f)

    print(f"Starting TTS generation for {len(segments)} segments")
    final_audio = AudioSegment.silent(duration=0)
    last_end_time = 0
    metadata = []

    for i, segment in enumerate(segments):
        # Generate TTS audio
        generator = tts(
            segment["text"], voice=voice,
            speed=1, split_pattern=r'\n+'
        )
        wav_list = [wav for _, _, wav in generator]
        if not wav_list:
            continue
        wav = np.concatenate(wav_list)

        sample_rate = 24000
        audio_duration = len(wav) / sample_rate
        original_duration = segment["end"] - segment["start"] # type: ignore

        # Add silence for gaps between segments
        segment_gap = (segment["start"] - last_end_time) * 1000 # type: ignore
        if segment_gap > 0:
            final_audio += AudioSegment.silent(duration=int(segment_gap))

        # Convert wav to int16 bytes
        wav_int16 = (wav * 32767).astype(np.int16).tobytes()
        final_audio += AudioSegment.from_file(
            io.BytesIO(wav_int16), format="raw", 
            frame_rate=sample_rate,
            channels=1, sample_width=2
        )
        last_end_time = segment["end"]
        metadata.append({"id": i, "duration": audio_duration, "original_duration": original_duration, "start": segment["start"], "end": segment["end"]})

    final_audio.export(str(out_audio), format="wav")
    with open(out_meta, "w") as f:
        json.dump(metadata, f)

    print(f"TTS audio generation completed. Output: {out_audio}")


def align_video(video: Path, tts: Path, metadata: Path, out: Path) -> None:
    with open(metadata, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    print(f"Processing {len(segments)} segments...")
    
    # Process each segment and collect file paths
    temp_files = []
    
    try:
        for i, segment in enumerate(segments):
            original_duration = segment['original_duration']
            tts_duration = segment['duration']
            
            # Calculate speed factor: how fast/slow should the video play?
            speed_factor = original_duration / tts_duration
            
            # Clamp to reasonable limits
            speed_factor = max(0.25, min(4.0, speed_factor))
            
            # Create temp file for this segment
            temp_file = tempfile.NamedTemporaryFile(suffix=f'_segment_{i}.mp4', delete=False)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Extract and speed-adjust this segment
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", str(segment['original_start']),
                "-t", str(original_duration),
                "-i", video
            ]
            
            # Add speed adjustment if needed
            if abs(speed_factor - 1.0) > 0.05:  # 5% tolerance
                cmd.extend(["-filter:v", f"setpts={1/speed_factor}*PTS"])
                print(f"Segment {i}: {'speeding up' if speed_factor > 1 else 'slowing down'} "
                      f"by {speed_factor:.2f}x ({original_duration:.1f}s → {tts_duration:.1f}s)")
            else:
                cmd.extend(["-c:v", "copy"])  # No re-encoding needed
                print(f"Segment {i}: no speed change needed")
            
            cmd.extend(["-an", temp_file.name])  # Remove audio, add output file
            
            # Run ffmpeg for this segment
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to process segment {i}: {result.stderr}")
        
        # Create concat file list
        concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for temp_path in temp_files:
            concat_file.write(f"file '{temp_path}'\n")
        concat_file.close()
        
        # Concatenate video segments
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video.close()
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_file.name,
            "-c", "copy", temp_video.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to concatenate segments: {result.stderr}")
        
        print("Video segments concatenated successfully")
        
        # Combine final video with TTS audio
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", temp_video.name,
            "-i", tts,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-y", str(out)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to combine video and audio: {result.stderr}")
        
        print(f"Final video saved to: {out}")
        
    finally:
        # Cleanup temp files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        
        try:
            os.unlink(concat_file.name)
            os.unlink(temp_video.name)
        except (OSError, NameError):
            pass


def extract_video_id(url: str):
    parsed = urlparse(url)
    if 'v=' in parsed.query:
        return parse_qs(parsed.query)['v'][0]
    else:
        # Fallback: use last part of path
        return Path(parsed.path).name or "unknown"

def check_not_revising(file_path: Path):
    if file_path.exists():
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line == REVISING_TAG:
                raise RuntimeError(f"❌ Error: {file_path} is still marked as revising.")

def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def copy(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)

def dub(url: str, out_dir: Path, mode: str, voice: str):
    video_id = extract_video_id(url)
    workdir = out_dir / video_id
    
    video = Target(workdir / "video.mp4", [], lambda out: get_video(url, out))
    audio = Target(workdir / "audio.wav", [], lambda out: get_audio(url, out))
    stt = Target(workdir / "transcript.json", [audio], lambda out: convert_stt(audio.out, out))
    transl = Target(workdir / "transcript.en.json", [stt], lambda out: translate_ar2en(stt.out, out))
    tts = Target(workdir / "tts.wav", [transl], lambda out: convert_tts(transl.out, out, workdir/"tts.meta.json", voice))
    final_audio = Target(workdir / "final.wav", [tts], lambda out: copy(tts.out, out))
    final_video = Target(workdir / "final.mp4", [video, tts], lambda out: align_video(video.out, tts.out, workdir/"tts.meta.json", out))

    if mode == "vid":
        targets = [video, audio, stt, transl, tts, final_video]
    else:
        targets = [audio, stt, transl, tts, final_audio]

    for target in targets:
        if target.is_done():
            del target
            continue

        if target.is_ready():
            target.gen()
            if target.needs_revision():
                return
        else:
            raise RuntimeError()

def main():
    parser = argparse.ArgumentParser(description="Taqreeb: Arabic to English Dubbing")
    parser.add_argument("url", help="Video URL (required for dub command)")
    parser.add_argument("--mode", "-m", choices=["vid", "aud"], default="vid", help="Output mode: vid (video) or aud (audio only)")
    parser.add_argument("--voice", "-v", default="em_eric", help="Voice for TTS")
    parser.add_argument("--out-dir", "-o", default="outs", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.url:
        print("❌ Error: URL is required for dubbing")
        print("Usage: python3 taqreeb.py 'https://youtube.com/watch?v=abc123'")
        sys.exit(1)

    dub(args.url, Path(args.out_dir), args.mode, args.voice)
