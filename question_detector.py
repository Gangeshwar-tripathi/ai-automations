import re


QUESTION_STARTS = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "can you",
    "could you",
    "would you",
    "do you",
    "does",
    "did",
    "is",
    "are",
    "will",
    "should",
)


def looks_like_question(text):

    text = text.strip().lower()

    if not text:
        return False

    # Explicit question
    if "?" in text:
        return True

    # Question-like sentence
    for start in QUESTION_STARTS:

        if text.startswith(start + " "):
            return True

    return False


def clean_question(text):

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()