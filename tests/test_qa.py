import os

from mini_rag.qa import RAGPipeline

DOCS = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_docs")


def _pipe():
    pipe = RAGPipeline()
    stats = pipe.ingest_folder(DOCS)
    assert stats["documents"] == 3
    return pipe


def test_ingest_counts_documents_and_chunks():
    pipe = RAGPipeline(chunk_size=50, overlap=10)
    stats = pipe.ingest_folder(DOCS)
    assert stats["documents"] == 3
    assert stats["chunks"] >= 3


def test_ask_returns_answer_with_citations():
    pipe = _pipe()
    out = pipe.ask("Where is the Taj Mahal?", k=2)
    assert "Agra" in out["answer"]
    assert any("taj-mahal.txt" in c for c in out["citations"])
    assert len(out["citations"]) == 2


def test_ask_empty_index_says_so():
    pipe = RAGPipeline()
    out = pipe.ask("anything")
    assert "no documents" in out["answer"].lower()


def test_evaluate_reports_metrics():
    pipe = _pipe()
    metrics = pipe.evaluate([
        ("What is photosynthesis?", "photosynthesis.txt"),
        ("Where is the Taj Mahal?", "taj-mahal.txt"),
        ("What is a Python list?", "python-basics.txt"),
    ], k=3)
    assert metrics["n"] == 3
    assert metrics["precision@k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_evaluate_empty_pairs():
    pipe = RAGPipeline()
    assert pipe.evaluate([]) == {"precision@k": 0.0, "mrr": 0.0, "n": 0}
