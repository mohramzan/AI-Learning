import ollama

response = ollama.embed(
    model="all-minilm",
    input="Semantic search finds documents by meaning."
)

embedding = response.embeddings[0]

print("Embedding model works!")
print("Number of dimensions:", len(embedding))
print("First five values:", embedding[:5])