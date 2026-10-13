"""Command-line interface for mini-rag.

    mini-rag ingest ./docs --index my.index
    mini-rag ask "What is photosynthesis?" --index my.index
    mini-rag eval --index my.index
"""
import argparse
import json
import sys

from .qa import RAGPipeline

DEFAULT_INDEX = "minirag.index"

SAMPLE_QA = [
    ("What is photosynthesis?", "photosynthesis.txt"),
    ("Where is the Taj Mahal?", "taj-mahal.txt"),
    ("What is a Python list?", "python-basics.txt"),
]


def _load_or_new(index, args):
    import os

    if os.path.exists(index):
        return RAGPipeline.load(index)
    return RAGPipeline(embedder=args.embedder, chunk_size=args.chunk_size,
                       overlap=args.overlap)


def cmd_ingest(args):
    pipe = RAGPipeline(embedder=args.embedder, chunk_size=args.chunk_size,
                       overlap=args.overlap)
    stats = pipe.ingest_folder(args.folder)
    pipe.save(args.index)
    print(f"Ingested {stats['documents']} documents -> {stats['chunks']} chunks.")
    print(f"Index saved to {args.index}")


def cmd_ask(args):
    pipe = _load_or_new(args.index, args)
    if len(pipe.store) == 0:
        print("Index is empty. Run `mini-rag ingest <folder>` first.", file=sys.stderr)
        sys.exit(1)
    out = pipe.ask(args.question, k=args.k)
    print("\nAnswer (extractive):\n" + out["answer"] + "\n")
    print("Sources:")
    for cite, score in zip(out["citations"], out["scores"]):
        print(f"  - {cite}  [score {score}]")


def cmd_eval(args):
    pipe = _load_or_new(args.index, args)
    if len(pipe.store) == 0:
        print("Index is empty. Run `mini-rag ingest <folder>` first.", file=sys.stderr)
        sys.exit(1)
    metrics = pipe.evaluate(SAMPLE_QA, k=args.k)
    print(json.dumps(metrics, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mini-rag",
                                     description="Minimal RAG Q&A over your documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Ingest a folder of .txt/.md files.")
    p.add_argument("folder")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=40)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Ask a question over the index.")
    p.add_argument("question")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=40)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="Evaluate retrieval on sample questions.")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=40)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
