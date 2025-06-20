# Agent RAG Local

This project implements a simple Retrieval-Augmented Generation agent that runs entirely locally. It processes PDF and Jupyter notebook course materials for the BUT SD programming modules and allows students to ask questions via a command line interface.

## Setup

```bash
python install_dependencies.py
```

## Usage

Build the vector index (only needed the first time or when supports change):

```bash
python main.py --build-index
```

Then start asking questions:

```bash
python main.py
```

## Project structure

- `preprocess/extract_text.py` – extract text from PDFs and notebooks
- `vector_store/build_qdrant_index.py` – chunk documents and build the Qdrant vector store
- `retrieval/search_chunks.py` – retrieve relevant chunks for a question
- `generation/generate_answer.py` – call the local Ollama model with retrieved context
- `main.py` – command line interface

All dependencies are listed in `requirements.txt`.
