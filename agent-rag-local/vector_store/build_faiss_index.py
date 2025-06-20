"""Build a FAISS vector index from plain text documents."""

from pathlib import Path
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS


class FaissIndexBuilder:
    def __init__(
        self,
        data_dir: Path | None = None,
        index_dir: Path | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parents[1] / "preprocess" / "output"
        self.index_dir = index_dir or Path(__file__).resolve().parent / "faiss_index"
        self.index_dir.mkdir(exist_ok=True)
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
        vector_store = FAISS.from_texts(chunks, self.model)
        vector_store.save_local(str(self.index_dir))
        print(f"Indexed {len(chunks)} chunks to {self.index_dir}")


if __name__ == "__main__":
    FaissIndexBuilder().build()
