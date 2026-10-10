
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Learning Python",
        "text": "Python is a beginner-friendly programming language."
    },
    {
        "id": "doc2",
        "title": "Software Development",
        "text": "Writing code and building software applications."
    },
    {
        "id": "doc3",
        "title": "Restarting an Application",
        "text": "Restart the application service if it stops responding."
    },
    {
        "id": "doc4",
        "title": "Healthy Eating",
        "text": "A balanced diet includes vegetables and fruits."
    },
    {
        "id": "doc5",
        "title": "Machine Learning",
        "text": "Machine learning systems learn patterns from data."
    },
    {
        "id": "doc6",
        "title": "Cloud Computing",
        "text": "Cloud services provide computing resources over a network."
    },
]


def main():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [doc["text"] for doc in DOCUMENTS]

    document_vectors = model.encode(
        texts,
        normalize_embeddings=True
    )

    print("Semantic Search")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nSearch: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a search query.")
            continue

        query_vector = model.encode(
            [query],
            normalize_embeddings=True
        )

        scores = cos_sim(
            query_vector,
            document_vectors
        )[0]

        ranked_indices = scores.argsort(
            descending=True
        )[:3].tolist()

        print("\nTop results:")

        for rank, index in enumerate(ranked_indices, start=1):
            doc = DOCUMENTS[index]
            score = float(scores[index])

            print(f"\n{rank}. {doc['title']} ({doc['id']})")
            print(f"   Similarity: {score:.4f}")
            print(f"   {doc['text']}")


if __name__ == "__main__":
    main()
