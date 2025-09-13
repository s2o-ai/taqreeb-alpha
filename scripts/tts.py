import json
from pathlib import Path
from argparse import ArgumentParser
from kokoro import KPipeline
from pydub import AudioSegment
import torch
import numpy as np

    parser = ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--output-dir", "-o", default="output")
    parser.add_argument("--voice", "-v", default="am_eric")

    args = parser.parse_args()

    file_path: Path = Path(args.file)
    output_dir = Path(args.output_dir)
    voice: str = args.voice

    print("Initializing TTS model")

    device = "gpu" if torch.cuda.is_available() else "cpu" # TODO: run TTS in GPU
    tts = KPipeline(lang_code="a")

    print("TTS model loaded successfully")

    with open(file_path, "r", encoding="utf-8") as f:
        segments: list[dict[str, str | float]] = json.load(f)

    print(f"Starting TTS generation for {len(segments)} segments")
    final_audio = AudioSegment.silent(duration=0)
    last_end_time = 0

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
        last_end_time = segment["end"]

    file_path = Path(output_dir) / "tts.wav"
    final_audio.export(str(file_path), format="wav")

    print(f"TTS audio generation completed. Output: {file_path}")
