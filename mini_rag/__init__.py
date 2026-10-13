"""mini-rag: a minimal, fully working RAG (retrieval-augmented generation)
Q&A system over your own documents.

Pipeline: ingest .txt/.md files -> chunk -> embed -> vector search ->
extractive answer with citations. No API key needed.
"""

from .chunking import Chunk, chunk_text
from .embeddings import TfidfEmbedder, get_embedder
from .store import VectorStore
from .qa import RAGPipeline

__all__ = ["Chunk", "chunk_text", "TfidfEmbedder", "get_embedder", "VectorStore", "RAGPipeline"]
__version__ = "0.1.0"
