"""
chat.py — AORUS MASTER 16 Interactive Q&A Entry Point
══════════════════════════════════════════════════════════
Loads the Vector Index and LLM, then enters an interactive Q&A loop.
Supports streaming output with real-time TTFT / TPS display.

Usage:
    python chat.py
    python chat.py --model models/Llama-3.2-3B-Instruct-Q5_K_M.gguf
    python chat.py --model models/your-model.gguf --chunks data/chunks.json
"""

import argparse
import json
import os
import sys

# Allow modules within src/ to import each other
sys.path.insert(0, os.path.dirname(__file__))

from vector_index import VectorIndex, SUPPORTED_MODELS, DEFAULT_MODEL
from retrieval_generate import (
    retrieve,
    build_context,
    generate_stream,
    load_llm,
    DEFAULT_MODEL_PATH,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="AORUS MASTER 16 RAG Interactive Q&A")
    parser.add_argument("--model",     default=DEFAULT_MODEL_PATH,    help="Path to GGUF model file")
    parser.add_argument("--chunks",    default="data/chunks.json",    help="Path to chunks JSON file")
    parser.add_argument("--emb",       default="data/embeddings.npy", help="Path to embeddings cache (.npy)")
    parser.add_argument("--emb-model", default=DEFAULT_MODEL,
                        help=(
                            f"Embedding model to use (default: {DEFAULT_MODEL})\n"
                            f"Supported shorthand aliases: {list(SUPPORTED_MODELS.keys())}"
                        ))
    parser.add_argument("--top-k",     type=int, default=5,           help="Retrieval top-k (default: 5)")
    args = parser.parse_args()

    # ── Load chunks ───────────────────────────────────────
    if not os.path.exists(args.chunks):
        print(f"[Error] Chunks file not found: {args.chunks} — run chunk_create.py first.")
        sys.exit(1)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    print(f"[✓] Loaded {len(chunks_data)} chunks")

    # Bilingual chunk detection
    bilingual_count = sum(1 for c in chunks_data if c.get("text_en"))
    if bilingual_count:
        print(f"[✓] Bilingual chunks detected: {bilingual_count} / {len(chunks_data)} "
              f"(supports Traditional Chinese × English mixed queries)")

    # ── Build vector index ────────────────────────────────
    # Supports shorthand aliases (e.g. --emb-model e5-base)
    emb_model_name = SUPPORTED_MODELS.get(args.emb_model, args.emb_model)
    index = VectorIndex(model_name=emb_model_name)
    index.build(chunks_data, emb_cache=args.emb)

    # ── Load LLM ─────────────────────────────────────────
    llm = load_llm(args.model)

    print("\n" + "═" * 55)
    print("  AORUS MASTER 16 AI Spec Assistant")
    print("  Type your question and press Enter. Type 'exit' to quit.")
    print("═" * 55)

    # ── Interactive loop ──────────────────────────────────
    while True:
        try:
            query = input("\n>>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Bye]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "bye", "q"}:
            print("[Bye]")
            break

        # Retrieval
        retrieved = retrieve(index, query, top_k=args.top_k)
        context   = build_context(retrieved)

        print("\n", end="", flush=True)

        # Streaming generation
        answer_parts: list[str] = []
        metrics: dict = {}

        for token, m in generate_stream(llm, query, context):
            if token:
                print(token, end="", flush=True)
                answer_parts.append(token)
            if m:
                metrics = m

        print()  # newline after answer

        if metrics:
            print(
                f"\nTFT={metrics.get('ttft_ms', '?')} ms  "
                f"TPS={metrics.get('tps', '?')} tok/s  "
                f"Tokens={metrics.get('total_tokens', '?')}  "
                f"Total={metrics.get('total_ms', '?')} ms"
            )

    # ── Cleanup ───────────────────────────────────────────
    if hasattr(llm, "close"):
        llm.close()


if __name__ == "__main__":
    main()