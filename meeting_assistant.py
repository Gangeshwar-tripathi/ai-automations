import threading

from meeting_listener import MeetingListener
from question_detector import (
    looks_like_question,
    clean_question,
)
from ai_engine import AIEngine


class MeetingAssistant:

    def __init__(
        self,
        answer_callback,
    ):

        self.answer_callback = (
            answer_callback
        )

        self.ai = AIEngine()

        self.transcript = []

        self.max_context = 8

        self.listener = MeetingListener(
            self.on_speech
        )

    def start(self):

        print(
            "Starting meeting assistant..."
        )

        self.listener.start()

    def stop(self):

        print(
            "Stopping meeting assistant..."
        )

        self.listener.stop()

    def on_speech(self, text):

        self.transcript.append(text)

        if len(self.transcript) > self.max_context:

            self.transcript.pop(0)

        print(
            f"Transcript: {text}"
        )

        if not looks_like_question(text):

            return

        question = clean_question(
            text
        )

        context = "\n".join(
            self.transcript[-6:]
        )

        print(
            f"QUESTION DETECTED: {question}"
        )

        thread = threading.Thread(
            target=self.generate_answer,
            args=(
                question,
                context,
            ),
            daemon=True,
        )

        thread.start()

    def generate_answer(
        self,
        question,
        context,
    ):

        try:

            answer = self.ai.answer(
                question,
                context,
            )

            self.answer_callback(
                question,
                answer,
            )

        except Exception as error:

            print(
                f"AI error: {error}"
            )