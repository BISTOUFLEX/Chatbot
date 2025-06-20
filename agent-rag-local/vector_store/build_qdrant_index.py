"""Build a Qdrant vector index from plain text documents."""

from pathlib import Path
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantIndexBuilder:
    def __init__(
        self,
        data_dir: Path | None = None,
        collection_name: str = "but_sd_supports",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parents[1] / "preprocess" / "output"
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(path=str(Path.home() / ".qdrant"))

    def _load_documents(self) -> List[str]:
        return [p.read_text() for p in self.data_dir.glob("*.txt")]

    def _chunk_documents(self, docs: Iterable[str]) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return [chunk for doc in docs for chunk in splitter.split_text(doc)]

    def build(self) -> None:
        docs = self._load_documents()
        chunks = self._chunk_documents(docs)

        embeddings = self.model.encode(chunks, show_progress_bar=True)

        if self.collection_name in [c.name for c in self.client.get_collections().collections]:
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            self.collection_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
        )

        points = [
            PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": chunks[i]})
            for i in range(len(chunks))
        ]
        self.client.upload_collection(collection_name=self.collection_name, points=points)
        print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    QdrantIndexBuilder().build()
