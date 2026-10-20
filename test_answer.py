"""Search, coverage threshold, and answer tests. Fake embeddings, so no AWS call."""
import faiss
import numpy as np

from core import answer, retrieve

DIM = 8
POLICY = {"title": "Mess", "owner_office": "Hostel Office", "version": "v2", "effective_date": "2025-01-01"}


def _index_for(texts, vectors):
    index = faiss.IndexFlatIP(DIM)
    index.add(np.asarray(vectors, dtype="float32"))
    chunks = [{"pos": i, "page": i + 1, "text": t} for i, t in enumerate(texts)]
    return index, chunks


def test_search_returns_only_chunks_from_its_own_index():
    index, chunks = _index_for(["LIB one", "LIB two", "LIB three"], np.eye(DIM)[:3])
    hits = retrieve.search(index, chunks, np.eye(DIM)[5], k=3)  # query points away from every row
    assert len(hits) == 3
    assert all(c["text"].startswith("LIB") for _, c in hits)


def test_refuses_below_threshold_without_calling_llm():
    index, chunks = _index_for(["hostel fees", "hostel rules", "hostel gate"], np.eye(DIM)[:3])
    calls = {"gen": 0}

    def embed(texts):
        return np.full((len(texts), DIM), 0.01, dtype="float32")  # weak match everywhere

    def gen(system, prompt):
        calls["gen"] += 1
        return "should not be used"

    result = answer.answer_question("football score?", {**POLICY, "owner_office": "Hall Office"},
                                    index, chunks, embed_fn=embed, generate_fn=gen, threshold=0.5)
    assert result["answered"] is False
    assert "Hall Office" in result["answer"]
    assert calls["gen"] == 0


def test_answer_lists_pages_of_retrieved_chunks():
    index, chunks = _index_for(["meals at 8am", "meals at 1pm", "no meals on Sunday"], np.eye(DIM)[:3])

    def embed(texts):
        return np.eye(DIM, dtype="float32")[[0] * len(texts)]

    def gen(system, prompt):
        assert "page 1" in prompt
        return "Breakfast is at 8am (page 1)."

    result = answer.answer_question("when is breakfast?", POLICY, index, chunks,
                                    embed_fn=embed, generate_fn=gen, threshold=0.1)
    assert result["answered"] is True
    assert result["answer"] == "Breakfast is at 8am (page 1)."
    assert result["pages"] == [1, 2, 3]


def test_prompt_contains_policy_details_and_question():
    chunks = [{"pos": 0, "page": 4, "text": "Rule text."}]
    prompt = answer.build_prompt("can I stay late?", POLICY, [(0.9, chunks[0])])
    assert "Mess" in prompt and "v2" in prompt and "2025-01-01" in prompt
    assert "page 4" in prompt
    assert prompt.endswith("Student question: can I stay late?")
