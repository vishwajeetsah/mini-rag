import os

from mini_rag.chunking import Chunk
from mini_rag.embeddings import TfidfEmbedder
from mini_rag.store import VectorStore


def _store():
    s = VectorStore(TfidfEmbedder())
    s.add([
        Chunk("a.txt", 0, "photosynthesis converts sunlight into energy in plants"),
        Chunk("b.txt", 0, "the taj mahal is a marble mausoleum in agra india"),
        Chunk("c.txt", 0, "python lists are ordered mutable collections"),
    ])
    return s


def test_search_returns_most_relevant_first():
    s = _store()
    results = s.search("marble mausoleum agra", k=3)
    assert results[0][0].doc_id == "b.txt"
    assert len(results) == 3


def test_search_scores_descend():
    s = _store()
    scores = [score for _, score in s.search("python lists", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_empty_store_search():
    s = VectorStore(TfidfEmbedder())
    assert s.search("anything") == []


def test_save_and_load_roundtrip(tmp_path):
    s = _store()
    path = os.path.join(str(tmp_path), "idx.pkl")
    s.save(path)
    s2 = VectorStore.load(path)
    assert len(s2) == 3
    results = s2.search("marble mausoleum agra", k=1)
    assert results[0][0].doc_id == "b.txt"
