OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

LLM_MODEL = "llama3.2:3b"
VISION_MODEL = "llava:7b"

WHISPER_MODEL = "base.en"

SAMPLE_RATE = 16000
CHANNELS = 1

# Put the audio device index here.
# Run:
# python3 -c "import sounddevice as sd; print(sd.query_devices())"
AUDIO_DEVICE = None

CHUNK_SECONDS = 5

NUM_CTX = 1024
NUM_PREDICT = 120