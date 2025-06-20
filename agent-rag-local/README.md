 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/agent-rag-local/README.md
index 0000000000000000000000000000000000000000..2eee713a636aae359717207ce82a96a9da87e7a0 100644
--- a//dev/null
+++ b/agent-rag-local/README.md
@@ -0,0 +1,33 @@
+# Agent RAG Local
+
+This project implements a simple Retrieval-Augmented Generation agent that runs entirely locally. It processes PDF and Jupyter notebook course materials for the BUT SD programming modules and allows students to ask questions via a command line interface.
+
+## Setup
+
+```bash
+python install_dependencies.py
+```
+
+## Usage
+
+Build the vector index (only needed the first time or when supports change):
+
+```bash
+python main.py --build-index
+```
+
+Then start asking questions:
+
+```bash
+python main.py
+```
+
+## Project structure
+
+- `preprocess/extract_text.py` – `Preprocessor` class for extracting text from supports
+- `vector_store/build_faiss_index.py` – `FaissIndexBuilder` class to build the vector store
+- `retrieval/search_chunks.py` – `Retriever` class to search for relevant chunks
+- `generation/generate_answer.py` – `AnswerGenerator` class wrapping the Ollama model
+- `main.py` – `LocalRAGAgent` command line interface
+
+All dependencies are listed in `requirements.txt`.
 
EOF
)
