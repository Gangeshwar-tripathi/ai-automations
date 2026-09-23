import threading
import tempfile
import os

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL,
    SAMPLE_RATE,
    CHANNELS,
    AUDIO_DEVICE,
    CHUNK_SECONDS,
)


class MeetingListener:

    def __init__(self, callback):

        self.callback = callback

        self.running = False

        self.whisper = None

        self.thread = None

    def load(self):

        print("Loading Whisper...")

        self.whisper = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )

        print("Whisper ready.")

    def start(self):

        if self.running:
            return

        if self.whisper is None:
            self.load()

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        self.running = False

    def _run(self):

        frames = int(
            CHUNK_SECONDS * SAMPLE_RATE
        )

        print("Meeting listener started.")

        while self.running:

            try:

                audio = sd.rec(
                    frames,
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    device=AUDIO_DEVICE,
                )

                sd.wait()

                if not self.running:
                    break

                self._transcribe(audio)

            except Exception as error:

                print(
                    f"Audio error: {error}"
                )

                self.running = False

    def _transcribe(self, audio):

        temp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        )

        path = temp.name

        temp.close()

        try:

            write(
                path,
                SAMPLE_RATE,
                audio,
            )

            segments, info = (
                self.whisper.transcribe(
                    path,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                )
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            if text:

                print(
                    f"MEETING: {text}"
                )

                self.callback(text)

        finally:

            try:
                os.remove(path)
            except OSError:
                pass