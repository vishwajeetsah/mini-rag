"""The RAG pipeline: ingest documents, answer questions, evaluate retrieval.

Answers are *extractive*: the most relevant sentence(s) from the retrieved
chunks, shown with citations. This keeps the system honest -- it cannot
hallucinate facts that are not in your documents.
"""
import os
import re

from .chunking import chunk_text
from .embeddings import get_embedder
from .store import VectorStore

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _best_sentences(chunk_text_, query, n=2):
    """Pick the n sentences in a chunk with the highest query-word overlap."""
    query_words = set(re.findall(r"\w+", query.lower()))
    sentences = [s.strip() for s in _SENT_SPLIT.split(chunk_text_) if s.strip()]
    if not sentences:
        return []
    scored = []
    for s in sentences:
        words = set(re.findall(r"\w+", s.lower()))
        scored.append((len(query_words & words), s))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for score, s in scored[:n] if score > 0] or [scored[0][1]]


class RAGPipeline:
    def __init__(self, embedder="tfidf", chunk_size=200, overlap=40):
        self.embedder_name = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = VectorStore(get_embedder(embedder))

    # ---- ingest ----
    def ingest_folder(self, folder):
        n_docs, n_chunks = 0, 0
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            chunks = chunk_text(text, doc_id=name,
                                chunk_size=self.chunk_size, overlap=self.overlap)
            self.store.add(chunks)
            n_docs += 1
            n_chunks += len(chunks)
        return {"documents": n_docs, "chunks": n_chunks}

    # ---- ask ----
    def ask(self, question, k=3, sentences_per_chunk=2):
        results = self.store.search(question, k=k)
        if not results:
            return {"answer": "I don't know -- no documents have been ingested yet.",
                    "citations": [], "scores": []}
        parts, citations, scores = [], [], []
        for chunk, score in results:
            for sent in _best_sentences(chunk.text, question, n=sentences_per_chunk):
                parts.append(sent)
            citations.append(f"{chunk.doc_id} (chunk {chunk.chunk_id})")
            scores.append(round(score, 4))
        answer = " ".join(parts)
        return {"answer": answer, "citations": citations, "scores": scores}

    # ---- evaluate ----
    def evaluate(self, qa_pairs, k=3):
        """qa_pairs: list of (question, expected_doc_id).

        Reports precision@k and MRR: does the right document show up
        in the top-k retrieved chunks?
        """
        if not qa_pairs:
            return {"precision@k": 0.0, "mrr": 0.0, "n": 0}
        hits, mrr = 0, 0.0
        for question, expected_doc in qa_pairs:
            results = self.store.search(question, k=k)
            rank = next((i for i, (c, _) in enumerate(results, 1)
                         if c.doc_id == expected_doc), None)
            if rank is not None:
                hits += 1
                mrr += 1.0 / rank
        n = len(qa_pairs)
        return {"precision@k": round(hits / n, 4), "mrr": round(mrr / n, 4), "n": n}

    # ---- persistence ----
    def save(self, path):
        self.store.save(path)

    @classmethod
    def load(cls, path):
        store = VectorStore.load(path)
        pipe = cls.__new__(cls)
        pipe.store = store
        pipe.embedder_name = "loaded"
        pipe.chunk_size = pipe.overlap = 0
        return pipe
