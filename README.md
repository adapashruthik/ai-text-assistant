# Local AI Document Q&A Assistant

A local-first AI application that answers questions about text documents using semantic search, embeddings, and Retrieval-Augmented Generation (RAG).

## Features

- Loads text documents from a local knowledge folder.
- Splits documents into overlapping chunks.
- Generates embeddings with Sentence Transformers.
- Retrieves relevant passages using cosine similarity.
- Uses a local Ollama language model to generate answers.
- Displays source document references.
- Includes unit tests for text processing, vector mathematics, prompt construction, and document loading.

## Technology Stack

- Python
- Sentence Transformers
- Ollama
- Requests
- unittest
- Git and GitHub

## Architecture

Documents → Chunking → Embeddings → Semantic Retrieval → Grounded Prompt → Local LLM → Answer

## Requirements

- Python and a virtual environment
- Ollama installed locally
- A compatible Ollama language model downloaded
- Internet access for the initial Python dependency and embedding-model downloads

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `python -m pip install -r requirements.txt`.
4. Download the embedding model on its first use.
5. Pull the configured Ollama model with `ollama pull llama3.2:3b`.
6. Place `.txt` files inside the `knowledge` directory.
7. Start Ollama and run `python final_rag_assistant.py`.

## Testing

Run `python -m unittest -v`.

## Limitations

- The example uses an in-memory document index.
- Only text files are supported by the current loader.
- Retrieval quality depends on the embedding model, document chunks, and query.
- The language model can still produce unsupported answers; results should be evaluated.
- Model downloads require disk space, and local inference performance depends on available hardware.

## Future Improvements

- Add PDF ingestion.
- Persist document embeddings.
- Add relevance thresholds and hybrid search.
- Add a web interface.
- Improve automated evaluation and source attribution.

## License

Add an appropriate license before publishing if you want others to reuse the project.