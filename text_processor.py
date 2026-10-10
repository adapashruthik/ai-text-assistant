
import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Count words in the cleaned text."""
    cleaned = clean_text(text)

    if not cleaned:
        return 0

    return len(cleaned.split())


def get_words(text: str) -> list[str]:
    """Return lowercase words without basic punctuation."""
    cleaned = clean_text(text).lower()
    return re.findall(r"\b[\w']+\b", cleaned)


if __name__ == "__main__":
    message = "  AI   developers build useful applications!  "

    print("Clean:", clean_text(message))
    print("Word count:", count_words(message))
    print("Words:", get_words(message))
