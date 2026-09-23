import time

from meeting_assistant import (
    MeetingAssistant,
)


def answer_received(
    question,
    answer,
):

    print("\n")
    print("=" * 60)

    print(
        "QUESTION:"
    )

    print(question)

    print("\nANSWER:")

    print(answer)

    print("=" * 60)


assistant = MeetingAssistant(
    answer_received
)

assistant.start()

print(
    "Meeting assistant is listening..."
)

print(
    "Press Ctrl+C to stop."
)

try:

    while True:

        time.sleep(1)

except KeyboardInterrupt:

    assistant.stop()

    print(
        "Stopped."
    )