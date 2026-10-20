"""Coverage check, prompt building, and answer generation for one policy."""
from typing import Callable

import numpy as np

from core import config, retrieve

SYSTEM_PROMPT = """You answer questions about ONE college policy for students.

Rules:
- Use only the policy excerpts provided. Do not use outside knowledge.
- Cite the page for each point, like (page 3).
- If the excerpts do not answer the question, say that the policy does not cover it and suggest contacting the office named in the policy.
- Do not make disciplinary, legal, or medical judgments.
- Answer in plain language, briefly.
"""


def is_covered(hits: list[tuple[float, dict]], threshold: float = config.SCORE_THRESHOLD) -> bool:
    """True if the best match is similar enough to the question."""
    return bool(hits) and hits[0][0] >= threshold


def build_prompt(question: str, policy: dict, hits: list[tuple[float, dict]]) -> str:
    excerpts = [
        f"[Excerpt {n}, page {chunk['page']}]\n{chunk['text']}"
        for n, (_, chunk) in enumerate(hits, start=1)
    ]
    return (
        f"Policy: {policy['title']}\n"
        f"Version: {policy.get('version') or 'not stated'}\n"
        f"Effective date: {policy.get('effective_date') or 'not stated'}\n\n"
        "Policy excerpts:\n\n" + "\n\n".join(excerpts) +
        f"\n\nStudent question: {question}"
    )


def answer_question(
    question: str,
    policy: dict,
    index,
    chunks: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
    generate_fn: Callable[[str, str], str],
    threshold: float = config.SCORE_THRESHOLD,
) -> dict:
    """Answer one question using one policy. Returns {"answered", "answer", "pages"}."""
    query_vec = embed_fn([question])[0]
    hits = retrieve.search(index, chunks, query_vec, config.TOP_K)

    if not is_covered(hits, threshold):
        office = policy.get("owner_office") or "the office responsible for this policy"
        return {
            "answered": False,
            "answer": f"This policy doesn't appear to cover that. Please contact {office}.",
            "pages": [],
        }

    text = generate_fn(SYSTEM_PROMPT, build_prompt(question, policy, hits))
    return {
        "answered": True,
        "answer": text,
        "pages": sorted({chunk["page"] for _, chunk in hits}),
    }
