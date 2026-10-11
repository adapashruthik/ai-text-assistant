
import requests
from sentence_transformers import SentenceTransformer

from rag_core import (
    KNOWLEDGE_BASE,
    retrieve_documents,
    build_prompt,
)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"


def ask_ollama(system_prompt, user_prompt):
    """Send a grounded prompt to the local Ollama model."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.1,
            },
        },
        timeout=180,
    )

    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]


def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    document_texts = [
        document["text"] for document in KNOWLEDGE_BASE
    ]

    document_vectors = embedding_model.encode(
        document_texts,
        normalize_embeddings=True,
    )

    print("RAG assistant ready.")
    print("Ask about the knowledge base or type 'exit'.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Assistant: Please enter a question.")
            continue

        try:
            documents = retrieve_documents(
                question,
                embedding_model,
                document_vectors,
                top_k=3,
            )

            if not documents:
                print("Assistant: No relevant documents were found.")
                continue

            system_prompt, user_prompt = build_prompt(
                question,
                documents,
            )

            print("\nRetrieved sources:")
            for document in documents:
                print(
                    f"- {document['title']} "
                    f"(ID: {document['id']}, "
                    f"similarity: {document['score']:.3f})"
                )

            print("\nGenerating answer...")
            answer = ask_ollama(system_prompt, user_prompt)

            print("\nAssistant:", answer)

        except requests.ConnectionError:
            print(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )
        except requests.Timeout:
            print("The model took too long. Try again or use a smaller model.")
        except requests.RequestException as error:
            print("Ollama request failed:", error)
        except (KeyError, ValueError) as error:
            print("Application error:", error)


if __name__ == "__main__":
    main()
