# mini-rag

A minimal, fully working **RAG (retrieval-augmented generation)** Q&A system over your own documents. Ingest a folder of `.txt`/`.md` files, ask questions in plain English, and get answers pulled straight from your documents — with citations. No API key, no cloud, no GPU needed.

Built as a learning project: every stage of a real RAG pipeline (chunking → embeddings → vector search → answer generation) is implemented from scratch in readable Python, so you can see exactly how it works.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Ingest your documents into an index
python -m mini_rag.cli ingest examples/sample_docs --index my.index
# Ingested 3 documents -> 3 chunks.

# 2. Ask questions
python -m mini_rag.cli ask "Where is the Taj Mahal?" --index my.index
```

Example output:

```
Answer (extractive):
The Taj Mahal is an ivory-white marble mausoleum on the southern bank
of the Yamuna river in Agra, India. ...

Sources:
  - taj-mahal.txt (chunk 0)  [score 0.7399]
  - python-basics.txt (chunk 0)  [score 0.6219]
```

```bash
# 3. Evaluate retrieval quality on sample questions
python -m mini_rag.cli eval --index my.index
# {"precision@k": 1.0, "mrr": 1.0, "n": 3}
```

## How it works

```
docs/ ──▶ chunk ──▶ embed ──▶ vector store ──▶ search ──▶ answer
          (200-word     (TF-IDF or      (numpy cosine     (top-k chunks →
           windows,      sentence-      similarity)       best sentences
           40 overlap)   transformers)                   + citations)
```

- **Chunking** (`mini_rag/chunking.py`): splits documents into overlapping word windows so facts straddling a boundary stay retrievable.
- **Embeddings** (`mini_rag/embeddings.py`): TF-IDF vectors by default (deterministic, zero downloads). Pass `--embedder sbert` for semantic embeddings via `sentence-transformers` (optional).
- **Vector store** (`mini_rag/store.py`): L2-normalised vectors in numpy; cosine similarity is a dot product. Save/load the index with pickle.
- **Q&A** (`mini_rag/qa.py`): *extractive* answers — the highest query-overlap sentences from the top-k chunks, each with a document citation and similarity score. Extractive by design: the system cannot hallucinate facts that aren't in your documents.
- **Eval** (`mini_rag/qa.py::evaluate`): precision@k and MRR over question → expected-document pairs, so you can measure whether retrieval actually finds the right doc.

## Project structure

```
mini-rag/
├── mini_rag/
│   ├── __init__.py
│   ├── chunking.py      # overlapping word-window chunking
│   ├── embeddings.py    # TF-IDF (default) + optional sentence-transformers
│   ├── store.py         # numpy vector store: search, save, load
│   ├── qa.py            # RAGPipeline: ingest, ask, evaluate
│   └── cli.py           # ingest / ask / eval commands
├── tests/               # 18 pytest tests
├── examples/sample_docs/
├── requirements.txt
└── .github/workflows/ci.yml
```

## Limitations (honest ones)

- TF-IDF matches keywords, not meaning — "car" won't match "automobile". Use the `sbert` backend for semantic search.
- Extractive answers quote sentences; they don't synthesise or reason across chunks.
- No re-ranking, hybrid search, or metadata filtering — deliberately, to keep the code readable.

## What I Learned

- **RAG is plumbing, not magic.** The "AI" part people demo is 10% of the system; chunking strategy, embedding choice, and retrieval quality decide whether answers are right.
- **Chunk overlap matters more than it looks.** Without overlap, a fact split across a boundary becomes unretrievable — a silent failure you'd never notice without eval.
- **TF-IDF is an underrated baseline.** On small corpora it retrieves surprisingly well, it's deterministic (great for tests), and it makes the value of semantic embeddings concrete when you compare.
- **Evaluation is the difference between a demo and a system.** precision@k / MRR turned "it feels right" into a number I could actually improve.
- **Extractive QA is a feature, not a compromise.** For a learning project, citing exact sentences beats fluent-but-possibly-hallucinated generation — and it teaches why grounding matters.

## License

MIT — see [LICENSE](LICENSE).
