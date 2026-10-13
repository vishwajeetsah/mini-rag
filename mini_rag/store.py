"""In-memory vector store with cosine-similarity search and persistence.

Vectors are L2-normalised by the embedder, so cosine similarity reduces to
a dot product.
"""
import pickle

import numpy as np


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.vectors = None  # shape (n_chunks, dim), float32

    def __len__(self):
        return len(self.chunks)

    def add(self, chunks):
        chunks = list(chunks)
        if not chunks:
            return
        vecs = self.embedder.encode([c.text for c in chunks])
        self.chunks.extend(chunks)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])

    def search(self, query, k=3):
        """Return [(Chunk, score)] for the top-k chunks, best first."""
        if not self.chunks:
            return []
        if k <= 0:
            raise ValueError("k must be positive")
        q = self.embedder.encode([query])[0]
        sims = self.vectors @ q
        top = np.argsort(-sims, kind="stable")[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectors": self.vectors,
                         "embedder": self.embedder}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        store = cls(payload["embedder"])
        store.chunks = payload["chunks"]
        store.vectors = payload["vectors"]
        return store
