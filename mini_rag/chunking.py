"""Split documents into overlapping word-based chunks.

Overlap exists so that a fact straddling a chunk boundary is still
retrievable in full from at least one chunk.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def chunk_text(text, doc_id, chunk_size=200, overlap=40):
    """Split *text* into chunks of ~chunk_size words with *overlap* words shared.

    Returns a list of Chunk. Empty/whitespace-only text yields [].
    """
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap

    chunks = []
    i, cid = 0, 0
    while i < len(words):
        piece = words[i : i + chunk_size]
        chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=" ".join(piece)))
        cid += 1
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks
