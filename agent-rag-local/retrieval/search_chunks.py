"""Retrieve relevant text chunks from a FAISS index."""

from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
import json
import faiss


class Retriever:
    def __init__(self, index_path: Path | None = None, metadata_path: Path | None = None) -> None:
        self.index_path = index_path or Path(__file__).resolve().parents[1] / "vector_store" / "index.faiss"
        self.metadata_path = metadata_path or Path(__file__).resolve().parents[1] / "vector_store" / "chunks.json"
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query: str, k: int = 3) -> List[str]:
        vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _D, I = self.index.search(vector, k)
        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]


if __name__ == "__main__":
    for c in Retriever().search("qu'est ce qu'une fonction ?"):
        print("--", c[:80])
