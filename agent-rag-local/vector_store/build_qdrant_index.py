from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

DATA_DIR = Path(__file__).resolve().parents[1] / "preprocess" / "output"
COLLECTION_NAME = "but_sd_supports"


def load_documents() -> List[str]:
    return [p.read_text() for p in DATA_DIR.glob("*.txt")]


def build_index():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [chunk for doc in docs for chunk in splitter.split_text(doc)]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    client = QdrantClient(path=str(Path.home() / ".qdrant"))
    if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE))

    points = [PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": chunks[i]}) for i in range(len(chunks))]
    client.upload_collection(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    build_index()
