"""Retrieve relevant text chunks from Qdrant."""

from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


class Retriever:
    def __init__(self, collection_name: str = "but_sd_supports") -> None:
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(path=str(Path.home() / ".qdrant"))

    def search(self, query: str, k: int = 3) -> List[str]:
        vector = self.model.encode([query])[0].tolist()
        results = self.client.search(collection_name=self.collection_name, query_vector=vector, limit=k)
        return [r.payload.get("text", "") for r in results]


if __name__ == "__main__":
    for c in Retriever().search("qu'est ce qu'une fonction ?"):
        print("--", c[:80])
