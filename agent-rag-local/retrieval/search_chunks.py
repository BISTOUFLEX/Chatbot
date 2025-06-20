"""Retrieve relevant text chunks from a FAISS index."""

from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS


class Retriever:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_dir = index_dir or Path(__file__).resolve().parents[1] / "vector_store" / "faiss_index"
        self.vector_store = FAISS.load_local(str(self.index_dir), self.model)

    def search(self, query: str, k: int = 3) -> List[str]:
        docs = self.vector_store.similarity_search(query, k=k)
        return [d.page_content for d in docs]


if __name__ == "__main__":
    for c in Retriever().search("qu'est ce qu'une fonction ?"):
        print("--", c[:80])
