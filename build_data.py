"""Build the search data for every policy listed in data/policies.csv.

    python -m core.build_data

Each run rebuilds all policies. Run it again after you change a PDF or the CSV.
"""
import csv
import json

import faiss
import numpy as np

from core import config
from core.bedrock import embed_texts
from core.ingest import chunk_pages, extract_pages

MIN_TEXT_CHARS = 500  # fewer characters than this usually means a scanned PDF


def build_one(row: dict, embed_fn) -> int:
    """Chunk, embed and save one policy. Returns the number of chunks."""
    pages = extract_pages(config.PDF_DIR / row["file"])
    if sum(len(p.strip()) for p in pages) < MIN_TEXT_CHARS:
        raise ValueError("Too little text. The PDF may be scanned.")

    chunks = chunk_pages(pages)
    vectors = np.asarray(embed_fn([c["text"] for c in chunks]), dtype="float32")

    # Vectors from Titan are already normalized, so inner product = cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    folder = config.STORAGE_DIR / row["id"]
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(folder / "index.faiss"))
    (folder / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return len(chunks)


def main() -> None:
    with open(config.POLICIES_CSV, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("id")]

    for row in rows:
        print(f"- {row['id']}: ", end="", flush=True)
        try:
            print(f"ok ({build_one(row, embed_texts)} chunks)")
        except Exception as exc:
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    main()
