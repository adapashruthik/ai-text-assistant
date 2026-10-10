
import ollama

MODEL = "llama3.2:3b"

messages = [
    {
        "role": "system",
        "content": (
            "You are a friendly AI and DevOps teacher. "
            "Explain concepts in simple English with "
            "practical examples."
        )
    }
]

print("AI Text Assistant")
print("Type 'exit' to stop.\n")

while True:
    try:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("AI: Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        # Add the user's question
        messages.append({
            "role": "user",
            "content": question
        })

        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages
            )

            answer = response["message"]["content"]

            # Save the assistant's answer
            messages.append({
                "role": "assistant",
                "content": answer
            })

            print("\nAI:", answer, "\n")

        except ollama.ResponseError as error:
            # Remove the failed user message
            messages.pop()

            print(f"\nOllama error: {error.error}")
            print("Check that the model is installed.\n")

        except (ollama.RequestError, ConnectionError, OSError) as error:
            messages.pop()

            print("\nCannot connect to Ollama.")
            print("Make sure Ollama is running.")
            print("Details:", error, "\n")

        except (KeyError, TypeError) as error:
            messages.pop()

            print("Unexpected response format:", error, "\n")

    except (KeyboardInterrupt, EOFError):
        print("\nAI: Goodbye!")
        break
