"""Utilities to extract text from PDF and notebook files."""

import os
from pathlib import Path

import nbformat
import fitz  # PyMuPDF


class Preprocessor:
    """Convert course supports into plain text files."""

    def __init__(self, supports_dir: Path | None = None, output_dir: Path | None = None) -> None:
        self.supports_dir = supports_dir or Path(__file__).resolve().parents[1] / "supports"
        self.output_dir = output_dir or Path(__file__).resolve().parent / "output"
        self.output_dir.mkdir(exist_ok=True)

    def _extract_pdf(self, path: Path) -> str:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    def _extract_notebook(self, path: Path) -> str:
        nb = nbformat.read(path, as_version=4)
        texts = [cell.get("source", "") for cell in nb.cells]
        return "\n".join(texts)

    def run(self) -> None:
        for root, _, files in os.walk(self.supports_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() == ".pdf":
                    text = self._extract_pdf(p)
                elif p.suffix.lower() == ".ipynb":
                    text = self._extract_notebook(p)
                else:
                    continue
                out_path = self.output_dir / f"{p.stem}.txt"
                out_path.write_text(text)
                print(f"Saved {out_path}")


if __name__ == "__main__":
    Preprocessor().run()
