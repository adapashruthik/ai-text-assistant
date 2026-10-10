
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


KNOWLEDGE_BASE = [
    {
        "id": "python",
        "text": "Python is a programming language used for automation, "
                "scripting, data analysis, and AI development."
    },
    {
        "id": "docker",
        "text": "Docker packages an application and its dependencies "
                "into containers for consistent execution."
    },
    {
        "id": "ml",
        "text": "Machine learning enables models to learn patterns "
                "from training data and make predictions."
    },
    {
        "id": "transformers",
        "text": "Transformers use attention mechanisms to process "
                "relationships between tokens in a sequence."
    },
    {
        "id": "embeddings",
        "text": "Text embeddings represent sentences or documents "
                "as numerical vectors for similarity search."
    },
    {
        "id": "cicd",
        "text": "CI/CD automates software integration, testing, "
                "building, and delivery or deployment."
    },
]


def build_search_index(model):
    texts = [item["text"] for item in KNOWLEDGE_BASE]

    vectors = model.encode(
        texts,
        normalize_embeddings=True
    )

    return vectors


def search_knowledge(query, model, vectors, top_k=3):
    if not query or not query.strip():
        return []

    query_vector = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cos_sim(query_vector, vectors)[0]
    indices = scores.argsort(descending=True)[:top_k].tolist()

    return [
        {
            "id": KNOWLEDGE_BASE[i]["id"],
            "text": KNOWLEDGE_BASE[i]["text"],
            "score": float(scores[i]),
        }
        for i in indices
    ]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = build_search_index(model)

    print("Semantic AI Assistant is ready.")
    print("Ask a question, or type 'exit' to quit.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "exit":
            break

        results = search_knowledge(query, model, vectors)

        if not results:
            print("Assistant: Please enter a question.")
            continue

        print("\nRetrieved information:")

        for result in results:
            print(f"\n[{result['id']}]")
            print(f"Similarity: {result['score']:.4f}")
            print(result["text"])


if __name__ == "__main__":
    main()
