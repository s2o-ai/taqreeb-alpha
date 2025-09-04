# -----------------------
# Configurable variables
# -----------------------
PYTHON       ?= python3
PIP          ?= $(PYTHON) -m pip
OUTPUT_DIR   ?= outputs
URL          ?=
MODE         ?= vid   # choices: vid | aud
VOICE        ?= em_eric

# Default requirements file
REQUIREMENTS ?= requirements.txt

# -----------------------
# Safety checks
# -----------------------
ifeq ($(strip $(URL)),)
$(error URL is not set. Usage: make dub url="https://youtube.com/watch?v=abc123" [mode=vid|aud] [voice=...] [output-dir=...])
endif

# -----------------------
# Derived variables
# -----------------------
VIDEO_ID     := $(shell basename $(URL) | cut -d= -f2)
WORKDIR      := $(OUTPUT_DIR)/$(VIDEO_ID)

VIDEO_FILE   := $(WORKDIR)/video.mp4
AUDIO_FILE   := $(WORKDIR)/audio.wav
STT_FILE     := $(WORKDIR)/transcript.json
TRANSL_FILE  := $(WORKDIR)/transcript.en.json
TTS_FILE     := $(WORKDIR)/tts.wav
FINAL_VIDEO   := $(WORKDIR)/final.mp4
FINAL_AUDIO  := $(WORKDIR)/final.wav

# -----------------------
# Targets
# -----------------------

.PHONY: all dub clean deps help

all: deps dub

## Run dubbing pipeline (url=... [mode=vid|aud] [voice=...] [output-dir=...])
dub: $(if $(filter $(MODE),vid),$(FINAL_VIDEO),$(FINAL_AUDIO))

# --- Step targets ---
$(VIDEO_FILE):
	@mkdir -p "$(WORKDIR)"
	$(PYTHON) scripts/get_video.py "$(URL)" --output-dir "$(WORKDIR)"

$(AUDIO_FILE): $(if $(filter $(MODE),vid),$(VIDEO_FILE))
	@mkdir -p "$(WORKDIR)"
	$(PYTHON) scripts/get_audio.py "$(URL)" --output-dir "$(WORKDIR)"

$(STT_FILE): $(AUDIO_FILE)
	$(PYTHON) scripts/stt.py "$(AUDIO_FILE)" --output-dir "$(WORKDIR)"

$(TRANSL_FILE): $(STT_FILE)
	$(PYTHON) scripts/ar2en.py "$(STT_FILE)" --output-dir "$(WORKDIR)"

$(TTS_FILE): $(TRANSL_FILE)
	$(PYTHON) scripts/tts.py "$(TRANSL_FILE)" --output-dir "$(WORKDIR)" --voice "$(VOICE)"

$(FINAL_VIDEO): $(TTS_FILE) $(TRANSL_FILE) $(STT_FILE) $(VIDEO_FILE)
	$(PYTHON) scripts/align_video.py "$(VIDEO_FILE)" "$(STT_FILE)" "$(TRANSL_FILE)" --output-dir "$(WORKDIR)"
	$(PYTHON) scripts/replace_audio.py "$(VIDEO_FILE)" "$(TTS_FILE)" --output-dir "$(WORKDIR)"

$(FINAL_AUDIO): $(TTS_FILE)
	cp "$(TTS_FILE)" "$(FINAL_AUDIO)"

## Clean outputs
clean:
	@echo "Cleaning output directory..."
	rm -rf "$(OUTPUT_DIR)"

## Install dependencies
deps:
	@echo "Installing dependencies..."
	$(PIP) install -r $(REQUIREMENTS)

## Show this help message
help:
	@echo "Available make targets:"
	@grep -E '^##' Makefile | sed 's/^## //'

