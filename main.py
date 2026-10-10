
from text_processor import clean_text, count_words, get_words


def generate_response(message: str) -> str:
    message = clean_text(message)

    if not message:
        return "Please enter a message."

    normalized = message.lower()

    if normalized in {"hi", "hello", "hey"}:
        return "Hello! How can I help you today?"

    if "machine learning" in normalized:
        return (
            "Machine learning enables computers to learn "
            "patterns from data."
        )

    if "transformer" in normalized:
        return (
            "Transformers use attention mechanisms to learn "
            "relationships between tokens."
        )

    words = get_words(message)

    return (
        f"You entered {count_words(message)} words. "
        f"Your main words are: {', '.join(words[:5])}"
    )


if __name__ == "__main__":
    print("AI Text Assistant — type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.strip().lower() == "exit":
            print("Assistant: Goodbye!")
            break

        print("Assistant:", generate_response(user_input))
