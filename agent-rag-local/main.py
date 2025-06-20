import argparse

from preprocess.extract_text import preprocess
from vector_store.build_qdrant_index import build_index
from retrieval.search_chunks import search
from generation.generate_answer import generate


def build():
    preprocess()
    build_index()


def chat():
    while True:
        try:
            question = input("Question > ")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip():
            continue
        chunks = search(question)
        context = "\n\n".join(chunks)
        prompt = (
            "Tu es un assistant pour les étudiants de BUT SD.\n"
            "Voici des extraits de cours pertinents :\n---\n"
            f"{context}\n---\n"
            f"Question : {question}\nRéponds clairement et précisément."
        )
        try:
            answer = generate(prompt)
        except Exception as e:
            print(f"Erreur génération : {e}")
            continue
        print(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true", help="rebuild vector index")
    args = parser.parse_args()
    if args.build_index:
        build()
    chat()


if __name__ == "__main__":
    main()
