"""Command line interface for the local RAG agent."""

import argparse

from preprocess.extract_text import Preprocessor
from vector_store.build_qdrant_index import QdrantIndexBuilder
from retrieval.search_chunks import Retriever
from generation.generate_answer import AnswerGenerator


class LocalRAGAgent:
    def __init__(self) -> None:
        self.preprocessor = Preprocessor()
        self.index_builder = QdrantIndexBuilder()
        self.retriever = Retriever()
        self.generator = AnswerGenerator()

    def build(self) -> None:
        self.preprocessor.run()
        self.index_builder.build()

    def answer(self, question: str) -> str:
        chunks = self.retriever.search(question)
        context = "\n\n".join(chunks)
        prompt = (
            "Tu es un assistant pour les étudiants de BUT SD.\n"
            "Voici des extraits de cours pertinents :\n---\n"
            f"{context}\n---\n"
            f"Question : {question}\nRéponds clairement et précisément."
        )
        return self.generator.generate(prompt)

    def chat(self) -> None:
        while True:
            try:
                question = input("Question > ")
            except (EOFError, KeyboardInterrupt):
                break
            if not question.strip():
                continue
            try:
                answer = self.answer(question)
            except Exception as e:  # pragma: no cover - runtime errors
                print(f"Erreur génération : {e}")
                continue
            print(answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true", help="rebuild vector index")
    args = parser.parse_args()

    agent = LocalRAGAgent()
    if args.build_index:
        agent.build()
    agent.chat()


if __name__ == "__main__":
    main()
