from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "but_sd_supports"
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path=str(Path.home() / ".qdrant"))


def search(query: str, k: int = 3) -> List[str]:
    vector = model.encode([query])[0].tolist()
    results = client.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=k)
    return [r.payload.get("text", "") for r in results]


if __name__ == "__main__":
    for c in search("qu'est ce qu'une fonction ?"):
        print("--", c[:80])
