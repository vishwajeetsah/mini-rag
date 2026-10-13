import numpy as np

from mini_rag.embeddings import TfidfEmbedder, get_embedder


def test_tfidf_vectors_are_l2_normalised():
    emb = TfidfEmbedder()
    vecs = emb.encode(["the cat sat", "dogs bark loudly", "quantum physics"])
    assert vecs.shape[0] == 3
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0)


def test_tfidf_is_deterministic():
    emb = TfidfEmbedder()
    a = emb.encode(["hello world", "foo bar"])
    b = emb.encode(["hello world", "foo bar"])
    assert np.allclose(a, b)


def test_similar_texts_score_higher():
    emb = TfidfEmbedder()
    vecs = emb.encode(["the cat sat on the mat", "the cat sat on the rug", "quantum field theory"])
    sim_close = float(vecs[0] @ vecs[1])
    sim_far = float(vecs[0] @ vecs[2])
    assert sim_close > sim_far


def test_get_embedder_factory():
    assert isinstance(get_embedder("tfidf"), TfidfEmbedder)
    try:
        get_embedder("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown embedder")
