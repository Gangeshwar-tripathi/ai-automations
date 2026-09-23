import json
import urllib.request

from config import (
    OLLAMA_URL,
    LLM_MODEL,
    NUM_CTX,
    NUM_PREDICT,
)


class AIEngine:

    def __init__(self):

        self.model = LLM_MODEL

    def answer(
        self,
        question,
        context="",
    ):

        prompt = f"""
You are a private meeting assistant.

Someone asked this question during a professional
technical meeting:

{question}

Recent meeting context:

{context}

Give a concise answer that the user can use to respond.

Format:

Definition:
One or two sentences.

Key Points:
- Important point
- Important point
- Important point

Suggested Response:
Give a natural response the user could say in the meeting.

Rules:
- Be accurate.
- Do not invent facts.
- If information is uncertain, say so.
- Keep it concise.
- Maximum 120 words.
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": -1,
            "options": {
                "temperature": 0.1,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
            },
        }

        data = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={
                "Content-Type":
                "application/json"
            },
            method="POST",
        )

        answer = ""

        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:

            for line in response:

                if not line:
                    continue

                try:

                    item = json.loads(
                        line.decode("utf-8")
                    )

                except json.JSONDecodeError:

                    continue

                answer += item.get(
                    "response",
                    "",
                )

        return answer