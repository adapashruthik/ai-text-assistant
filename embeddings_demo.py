
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "I love programming.",
    "I enjoy writing code.",
    "The weather is rainy.",
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

print("Number of sentences:", len(sentences))
print("Vector dimensions:", embeddings.shape[1])

for sentence, vector in zip(sentences, embeddings):
    print("\nText:", sentence)
    print("First five coordinates:", vector[:5])

print("\nSentence 1 vs Sentence 2:",
      round(float(cos_sim(embeddings[0], embeddings[1])), 4))

print("Sentence 1 vs Sentence 3:",
      round(float(cos_sim(embeddings[0], embeddings[2])), 4))
