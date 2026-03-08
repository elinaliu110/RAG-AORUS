# AORUS MASTER 16 — AI Hardware Spec Assistant (RAG)

>  [Benchmark Report](docs/benchmark_report_en.md)

A **pure Python RAG system** for answering GIGABYTE AORUS MASTER 16 AM6H product specifications.
Designed for consumer-grade laptops with ≤ 4 GB VRAM, with no dependency on LangChain or LlamaIndex.

---

## ✨ Key Features

- **Bilingual queries** — Traditional Chinese × English mixed input supported
- **Pure Python RAG** — No LangChain / LlamaIndex; custom Chunking, Retrieval, Generation
- **Dual-path retrieval** — Key filter (exact) + Vector search (semantic), minimising latency
- **Streaming output** — Real-time token streaming with TTFT / TPS measurement
- **4 GB VRAM compliant** — All tested models fit within consumer GPU constraints
- **Managed with `uv`** — Fast, reproducible Python environment

---

## 📁 Project Structure

```
aorus-rag/
├── README.md                      # This file (English)
├── README_ZH.md                   # 繁體中文說明
├── pyproject.toml                 # uv environment & dependencies
├── .gitignore
│
├── src/
│   ├── chunk_create.py            # specs.csv → bilingual chunks JSON
│   ├── vector_index.py            # Embedding index + exact key lookup
│   ├── retrieval_generate.py      # Filter extraction, retrieval, streaming LLM
│   ├── benchmark.py               # Quantitative evaluation (Hit Rate / TTFT / TPS)
│   └── run_main.py                # Interactive Q&A entry point
│
├── data/
│   ├── specs.csv                  # Raw AORUS MASTER 16 specification sheet
│   ├── chunks.json                # Bilingual chunks (auto-generated)
│   ├── embeddings.npy             # Embedding cache (auto-generated, gitignored)
│   └── benchmark_cases.json       # Evaluation test cases (10 queries)
│
├── models/                        # GGUF model files (gitignored — download separately)
│   └── README.md
│
├── results/                       # Benchmark outputs (PNG charts + JSON)
│   └── README.md
│
├── docs/
│   ├── benchmark_report_en.md     # Full benchmark analysis (English)
│   └── benchmark_report.md        # 完整評測報告（繁體中文）
│
└── scripts/
    └── download_model.py          # GGUF model download helper
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/aorus-rag.git
cd aorus-rag
uv sync
```

### 2. Download a Model

```bash
# Recommended (highest accuracy — 91.5% hit rate)
uv run python scripts/download_model.py --model llama-3.2-3b-q5

# Speed-optimised alternative
uv run python scripts/download_model.py --model llama-3.2-3b-q4

# List all available models
uv run python scripts/download_model.py --list
```

### 3. Build the Vector Index

```bash
uv run python src/vector_index.py \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy
```

> **Note:** If you regenerate `chunks.json` via `chunk_create.py`, always rebuild embeddings with `--force`.

### 4. Interactive Q&A

```bash
uv run python src/run_main.py \
    --model  models/Llama-3.2-3B-Instruct-Q5_K_M.gguf \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy
```

**Example queries:**
```
>>> AORUS MASTER 16 支援哪些作業系統？
>>> What wireless connectivity does the AORUS MASTER 16 support?
>>> BZH 和 BXH 的 GPU 功耗差異在哪裡？
>>> Which model should I choose between BYH and BXH for gaming?
```

### 5. Run Benchmark

```bash
uv run python src/benchmark.py \
    --model  models/Llama-3.2-3B-Instruct-Q5_K_M.gguf \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy \
    --cases  data/benchmark_cases.json \
    --out    results/benchmark_results_Llama-Q5.json
```

---

## 🏗 System Architecture

```
User Query (ZH / EN / Mixed)
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage C-1 · Filter Extraction           │
│  ├─ extract_product_filter()             │  → BZH / BYH / BXH / None
│  └─ extract_key_filter()                 │  → spec key alias match
└──────────────────┬───────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Key Filter hit?   │
         └──┬─────────────┬───┘
         Yes│             │No
            ▼             ▼
    Exact chunk       Vector Search
    retrieval         (multilingual-MiniLM cosine)
    (no encoding)
            │             │
            └──────┬───────┘
                   ▼
        build_context()   ← bilingual text, max 1400 tokens
                   │
                   ▼
        llama.cpp GGUF inference
        Streaming · TTFT / TPS measurement
                   │
                   ▼
            Answer (ZH or EN)
```

### Bilingual Chunk Design

Each chunk's `text` field combines Chinese and English for cross-lingual embedding:

```json
{
  "text_zh": "AORUS MASTER 16 BZH 的無線通訊支援 WIFI 7 (802.11be 2x2)，LAN: 1G，Bluetooth v5.4。",
  "text_en": "AORUS MASTER 16 BZH wireless connectivity supports WiFi 7 (802.11be 2x2), LAN 1G, Bluetooth 5.4, wireless, connectivity.",
  "text":    "<zh> / <en>"
}
```

This resolves cross-lingual retrieval failures where English queries could not match Chinese-only chunks.

---

## 📊 Model Benchmark Summary (CPU-only)

All tests run on CPU. GPU results would reduce TTFT to ~5–15 s range.

| Model | Hit Rate | Avg TTFT | TPS | RAM Peak |
|-------|:--------:|:--------:|:---:|:--------:|
| **Llama-3.2-3B Q5_K_M** ✅ | **91.5%** | 127,845 ms | 2.2 | 3,848 MB |
| Llama-3.2-3B Q4_K_M | 84.0% | **73,022 ms** | **2.6** | 4,858 MB |
| Qwen2.5-3B Q5_K_M | 82.5% | 145,409 ms | 2.2 | **3,506 MB** |
| Qwen2.5-3B Q4_K_M | 77.0% | 81,512 ms | 2.6 | 4,447 MB |
| Phi-4-mini Q4_K_M | 69.0% | 96,480 ms | 2.2 | 5,283 MB |
| Phi-4-mini Q5_K_M | 69.0% | 163,269 ms | 1.9 | 4,624 MB |

**Recommended:** `Llama-3.2-3B-Instruct-Q5_K_M` — highest accuracy, RAM well within 4 GB limit.

> 📄 Full analysis: [docs/benchmark_report_en.md](docs/benchmark_report_en.md)

---

## 🔧 Model Selection Rationale (4 GB Constraint)

| Scenario | Model | Reason |
|----------|-------|--------|
| **Accuracy-first** (default) | Llama-3.2-3B Q5_K_M | 91.5% hit rate, 3.8 GB RAM |
| **Speed-first** | Llama-3.2-3B Q4_K_M | 43% faster TTFT, 84% accuracy |
| **RAM-minimal** | Qwen2.5-3B Q5_K_M | 3.5 GB RAM, but Q6 multi-field EN weakness |
| ❌ Not recommended | Phi-4-mini (both) | 69% accuracy, highest RAM, hallucination on comparisons |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `llama-cpp-python` | GGUF inference engine (CPU + GPU) |
| `sentence-transformers` | Multilingual embedding model |
| `numpy` | Embedding vector operations |
| `psutil` | CPU / RAM monitoring during benchmark |
| `matplotlib` | Benchmark chart generation |

---

## 📝 Notes

- **No GPU data available** in current benchmarks — all results are CPU-only.
  GPU inference is expected to reduce TTFT to ~5–15 seconds.
- **`data/embeddings.npy`** is gitignored. Rebuild after any chunk changes:
  ```bash
  uv run python src/vector_index.py --chunks data/chunks.json --emb data/embeddings.npy --force
  ```
- **`models/`** is gitignored. Use `scripts/download_model.py` to fetch GGUF files.
