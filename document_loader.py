
from pathlib import Path


def load_text_files(folder):
    """Load .txt documents from a folder."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Knowledge folder not found: {folder}"
        )

    documents = []

    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text,
            })

    return documents


def chunk_text(text, chunk_size=700, overlap=100):
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_chunks(documents, chunk_size=700, overlap=100):
    """Create source-linked chunks from loaded documents."""
    chunks = []

    for document in documents:
        pieces = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, piece in enumerate(pieces, start=1):
            chunks.append({
                "id": f"{document['source']}:{index}",
                "source": document["source"],
                "text": piece,
            })

    return chunks


if __name__ == "__main__":
    docs = load_text_files("knowledge")
    chunks = build_chunks(docs)

    print("Documents loaded:", len(docs))
    print("Chunks created:", len(chunks))

    for chunk in chunks:
        print(chunk["id"], "->", chunk["text"][:80])
