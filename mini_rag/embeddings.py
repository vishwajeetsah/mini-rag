"""Embedding backends.

The default ``tfidf`` backend needs no downloads and is fully
deterministic -- ideal for learning and tests. The optional ``sbert``
backend uses sentence-transformers for higher-quality semantic vectors
when the package (and its model download) is available.
"""
import numpy as np


class TfidfEmbedder:
    """L2-normalised TF-IDF vectors. Fitted on the first texts it sees."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, texts):
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts):
        if not self._fitted:
            self.fit(texts)
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return mat / norms


class SbertEmbedder:
    """Semantic embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, texts):  # noqa: ARG002 - API compatibility
        return self

    def encode(self, texts):
        vecs = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vecs / norms


def get_embedder(name="tfidf"):
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "sbert":
        return SbertEmbedder()
    raise ValueError(f"unknown embedder: {name!r} (choose 'tfidf' or 'sbert')")
