"""Core functionality for the private meeting assistant.

This module intentionally contains no Cocoa UI code.  The UI layer is
responsible only for buttons, windows and displaying results.
"""

import base64
import json
import os
import subprocess
import tempfile
import urllib.request

from faster_whisper import WhisperModel

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
VISION_MODEL = "llava:7b"
NUM_CTX = 1024
NUM_PREDICT = 120
TEMPERATURE = 0.1
SAMPLE_RATE = 16000

MEETING_CONTEXT = """
You are a technical meeting assistant for a Python Backend Developer
with 4+ years of professional experience.

The developer works primarily with:
- Python
- Advanced Python and OOP
- FastAPI
- REST APIs
- Microservices
- Multithreading and multiprocessing
- AsyncIO and async/await
- SQL and databases
- Redis
- Docker
- CI/CD
- Backend system design

Answer at the level expected from an experienced Python backend developer.

Behavior:
- Give technically accurate answers.
- Prefer practical production-oriented explanations.
- Do not give overly basic or generic answers.
- Keep simple questions concise.
- Explain trade-offs when relevant.
- For debugging, explain likely cause, how to verify it, and the fix.
- For architecture questions, explain the approach and important trade-offs.
- For coding questions, provide clean practical Python code.
- Do not invent project-specific information.
- If information is missing, clearly state the assumption.
- If the question is ambiguous, explain what needs clarification.

Meeting goal:
Help the developer understand what is being asked and provide a
natural response they can say during a technical meeting.
""".strip()


class AIService:
    def __init__(self, ollama_url=OLLAMA_URL, model=LLM_MODEL,
                 vision_model=VISION_MODEL):
        self.ollama_url = ollama_url
        self.model = model
        self.vision_model = vision_model

    def _request(self, payload, timeout=120):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def answer_stream(self, question, context=""):
        prompt = f"""
        {MEETING_CONTEXT}

Question:
{question}

Recent context:
{context}

Give a concise answer the user can use in a professional technical interview .
Suggested Response:
A natural human response the user can say in the meeting.

Rules:
- Be accurate based on 4+ years of experience.
- give me the method to or function to solve the problem.
- Maximum 200 words.
- Python code should be in one markdown code block.
""".strip()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": -1,
            "options": {
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            for line in response:
                if not line:
                    continue
                try:
                    item = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                chunk = item.get("response", "")
                if chunk:
                    yield chunk

    def answer(self, question, context=""):
        return "".join(self.answer_stream(question, context))

    def analyze_screen(self, image_path):
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("ascii")

        prompt = """
Analyze this meeting screenshot.

Identify the main question, task, error, code, or technical topic visible
on the screen that the user may need to respond to.

Return only useful factual context for another AI model.
If a question is visible, quote the question accurately.
If there is code or an error, include the important part.
Do not invent information that is not visible.
""".strip()

        payload = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 300,
            },
        }
        result = self._request(payload, timeout=180)
        return result.get("response", "").strip()

    def answer_from_screen(self, screen_context):
        prompt = f"""
You are a private meeting assistant.

The following information was extracted from a screenshot by a vision model:

--- SCREEN CONTEXT ---
{screen_context}
--- END SCREEN CONTEXT ---

Give the user a concise answer they can use in a professional technical  interview.

Suggested Response:
A natural human response the user can say in the interview .

Rules:
- Do not invent facts.
- Maximum 400 words.
- 
""".strip()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
            },
        }
        result = self._request(payload, timeout=120)
        return result.get("response", "").strip()

    def capture_and_answer(self):
        path = None
        resized_path = None
        try:
            fd, path = tempfile.mkstemp(
                suffix=".png", prefix="meeting_screen_"
            )
            os.close(fd)
            subprocess.run(
                ["/usr/sbin/screencapture", "-x", "-m", path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            resized_path = path + ".small.png"
            subprocess.run(
                ["/usr/bin/sips", "-Z", "1600", path, "--out", resized_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            image_path = resized_path if os.path.exists(resized_path) else path
            context = self.analyze_screen(image_path)
            if not context:
                raise RuntimeError("LLaVA did not return any screen analysis.")
            return self.answer_from_screen(context), context
        finally:
            for temp_path in (resized_path, path):
                if temp_path:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass


class AudioService:
    def __init__(self, model_name="base.en"):
        self.model_name = model_name
        self.model = None

    def load(self):
        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
        )
        return self.model

    def transcribe(self, audio_data, sample_rate=SAMPLE_RATE):
        if self.model is None:
            self.load()
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        try:
            from scipy.io.wavfile import write
            write(temp_path, sample_rate, audio_data)
            segments, _ = self.model.transcribe(
                temp_path,
                beam_size=1,
                language="en",
                vad_filter=True,
            )
            return " ".join(
                segment.text.strip() for segment in segments
            ).strip()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
