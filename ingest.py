"""Read a policy PDF and split it into chunks. Every chunk remembers its page number."""
import re
from pathlib import Path

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import config

# Invisible characters that PDFs sometimes leave in extracted text
ZERO_WIDTH = re.compile(r"[​‌‍﻿]")


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the text of each page, in order. Page 1 is index 0."""
    reader = pypdf.PdfReader(str(pdf_path))
    return [ZERO_WIDTH.sub("", page.extract_text() or "") for page in reader.pages]


def chunk_pages(pages: list[str]) -> list[dict]:
    """Split each page on its own, so every chunk comes from exactly one page.

    Returns [{"pos": 0, "page": 1, "text": "..."}, ...]. `pos` is the chunk's row
    number, which must match its row in the FAISS index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_no, text in enumerate(pages, start=1):
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append({"pos": len(chunks), "page": page_no, "text": piece})
    return chunks
