import json
from argparse import ArgumentParser
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("--output-dir", "-o", default="output")
args = parser.parse_args()

file_path: Path = Path(args.file)
output_dir = Path(args.output_dir)

MODEL_ID = "Helsinki-NLP/opus-mt-ar-en"

print(f"Loading translation model: {MODEL_ID}")

tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
model = MarianMTModel.from_pretrained(MODEL_ID)

print("Translation model loaded successfully")


with open(file_path, "r") as f:
    segments: list[dict[str, str | float]] = json.load(f)

print(f"Starting Arabic to English translation for {len(segments)} segments")

# Create a copy of the transcript for translation
texts = [segment["text"] for segment in segments]

print("Tokenizing and translating texts...")
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
translated_tokens = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id)
translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

# Reconstruct segments with translations
text_idx = 0
for segment in segments:
    if not segment["text"].strip(): # type: ignore
        continue
    translation = translated_texts[text_idx] if text_idx < len(translated_texts) else segment["test"]
    segment["text"] = translation
    text_idx += 1

print("Writing translated segments")
with open(output_dir / "transcript.en.json", "w") as f:
    json.dump(segments, f)

print("Translation completed successfully")
