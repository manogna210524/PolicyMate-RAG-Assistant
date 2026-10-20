"""Chunking tests. No AWS calls."""
from core import config, ingest


def test_chunks_keep_their_page_number():
    pages = ["Alpha rule. " * 300, "Beta rule. " * 20]  # page 1 splits into several chunks
    chunks = ingest.chunk_pages(pages)
    assert [c["pos"] for c in chunks] == list(range(len(chunks)))
    assert {c["page"] for c in chunks} == {1, 2}
    assert all(c["page"] == 1 for c in chunks if "Alpha" in c["text"])
    assert all(c["page"] == 2 for c in chunks if "Beta" in c["text"])


def test_chunks_are_not_longer_than_chunk_size():
    chunks = ingest.chunk_pages(["Sentence number. " * 400])
    assert len(chunks) > 1
    assert all(len(c["text"]) <= config.CHUNK_SIZE for c in chunks)


def test_empty_pages_give_no_chunks():
    assert ingest.chunk_pages(["", "   "]) == []
