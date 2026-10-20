"""Load the policy list and search one policy at a time."""
import csv
import json

import faiss
import numpy as np

from core import config


def list_policies() -> list[dict]:
    """Policies from policies.csv that have been built (their index file exists)."""
    with open(config.POLICIES_CSV, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("id")]
    return [
        row for row in rows
        if (config.STORAGE_DIR / row["id"] / "index.faiss").exists()
    ]


def load(policy_id: str) -> tuple[faiss.Index, list[dict]]:
    """Return one policy's index and its chunks. chunks[i] matches index row i."""
    folder = config.STORAGE_DIR / policy_id
    index_path = folder / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"No data for '{policy_id}'. Run: python -m core.build_data")
    index = faiss.read_index(str(index_path))
    chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks


def search(index: faiss.Index, chunks: list[dict], query_vec: np.ndarray,
           k: int = config.TOP_K) -> list[tuple[float, dict]]:
    """Return up to k (score, chunk) pairs, best first. Only this policy's chunks can appear."""
    if index.ntotal == 0:
        return []
    k = min(k, index.ntotal)
    scores, ids = index.search(query_vec.reshape(1, -1).astype("float32"), k)
    return [(float(score), chunks[int(row)]) for score, row in zip(scores[0], ids[0]) if row >= 0]
