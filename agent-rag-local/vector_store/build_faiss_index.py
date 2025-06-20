"""Build a FAISS vector index from plain text documents."""

from pathlib import Path
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss


class FaissIndexBuilder:
    def __init__(
        self,
        data_dir: Path | None = None,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parents[1] / "preprocess" / "output"
        self.index_path = index_path or Path(__file__).resolve().parent / "index.faiss"
        self.metadata_path = metadata_path or Path(__file__).resolve().parent / "chunks.json"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_documents(self) -> List[str]:
        return [p.read_text() for p in self.data_dir.glob("*.txt")]

    def _chunk_documents(self, docs: Iterable[str]) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return [chunk for doc in docs for chunk in splitter.split_text(doc)]

    def build(self) -> None:
        docs = self._load_documents()
        chunks = self._chunk_documents(docs)

        embeddings = self.model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
        emb_array = np.array(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(emb_array.shape[1])
        index.add(emb_array)
        faiss.write_index(index, str(self.index_path))

        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f)
        print(f"Indexed {len(chunks)} chunks to {self.index_path}")


if __name__ == "__main__":
    FaissIndexBuilder().build()
