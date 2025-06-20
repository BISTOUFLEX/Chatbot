import os
import nbformat
import fitz  # PyMuPDF
from pathlib import Path

SUPPORTS_DIR = Path(__file__).resolve().parents[1] / "supports"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_pdf(path: Path) -> str:
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_notebook(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    texts = [cell.get("source", "") for cell in nb.cells]
    return "\n".join(texts)


def preprocess():
    for root, _, files in os.walk(SUPPORTS_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() == ".pdf":
                text = extract_pdf(p)
            elif p.suffix.lower() == ".ipynb":
                text = extract_notebook(p)
            else:
                continue
            out_path = OUTPUT_DIR / f"{p.stem}.txt"
            out_path.write_text(text)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    preprocess()
