# AORUS MASTER 16 — AI Spec Assistant (llama-cpp CPU Edition)

This project implements a RAG-based spec Q&A system using llama-cpp quantized models on CPU (Google Colab).

---
<!-- 
## Project Structure

```
aorus-rag/
├── README.md                      
├── pyproject.toml                 # uv environment & dependencies
├── .gitignore
│
├── src/
│   ├── chunk_create.py            # specs.csv → bilingual chunks JSON
│   ├── vector_index.py            # Embedding index + exact key lookup
│   ├── retrieval_generate.py      # Filter extraction, retrieval, streaming LLM
│   ├── benchmark.py               # Quantitative evaluation (Hit Rate / TTFT / TPS)
│   └── chat.py                    # Interactive Q&A entry point
│
├── data/
│   ├── specs.csv                  # Raw AORUS MASTER 16 specification sheet
│   ├── chunks.json                # Bilingual chunks
│   ├── embeddings.npy             # Embedding cache
│   └── benchmark_cases.json       # Evaluation test cases (10 queries: 5 ZH + 5 EN)
│
├── models/                        # GGUF model files
│
├── results/                       # Benchmark outputs (PNG charts + Ans JSON)
│   ├── Llama-3.2-3B_benchmark/
│   ├── Phi-4-mini_benchmark/
│   └── Qwen2.5-3B_benchmark/
│
├── docs/
│   └── benchmark_report.md     # benchmark analysis
│
└── scripts/
    └── download_model.py       # GGUF model download helper
``` -->

## Quick Start

### Prerequisites
- Python 3.11+
- `!pip install uv -q`

### 1. Clone & Install

```bash
!git clone https://github.com/elinaliu110/aorus-rag.git
%cd aorus-rag

# Install all project dependencies from pyproject_vllm.toml
!uv sync
```

### 2. Download a Model

```bash
# Recommended (highest accuracy — 91.5% hit rate)
!uv run python scripts/download_model.py --model llama-3.2-3b-q5

# Speed-optimised alternative
!uv run python scripts/download_model.py --model llama-3.2-3b-q4

# List all available models
!uv run python scripts/download_model.py --list
```

### 3. Create Chunks (if not yet done)

```bash
!uv run python src/chunk_create.py \
    --input  data/specs.csv \
    --output data/chunks.json
```

### 4. Build the Vector Index

```bash
!uv run python src/vector_index.py \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy
```

### 5. Interactive Q&A

```bash
!uv run python src/chat_llamacpp.py \
    --model  models/Llama-3.2-3B-Instruct-Q5_K_M.gguf \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy
```

**Example queries:**
```
>>> AORUS MASTER 16 支援哪些作業系統？
>>> What is the AORUS MASTER 16 BXH battery capacity?
```

### 6. Run Benchmark

```bash
!uv run python src/benchmark_vllm.py \
    --model  models/Llama-3.2-3B-Instruct-Q5_K_M.gguf \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy \
    --cases  data/benchmark_cases.json \
    --out    results/benchmark_results_Llama-Q5.json
```

## Model Benchmark Summary (CPU)

All tests run on CPU.

| Model | Hit Rate | Avg TTFT | TPS |
|-------|:--------:|:--------:|:---:|
| **Llama-3.2-3B Q5_K_M** | **91.5%** | 127,845 ms | 2.2 |
| Llama-3.2-3B Q4_K_M | 84.0% | **73,022 ms** | **2.6** |
| Qwen2.5-3B Q5_K_M | 82.5% | 145,409 ms | 2.2 |
| Qwen2.5-3B Q4_K_M | 77.0% | 81,512 ms | 2.6 |
| Phi-4-mini Q4_K_M | 69.0% | 96,480 ms | 2.2 |
| Phi-4-mini Q5_K_M | 69.0% | 163,269 ms | 1.9 |

**Recommended:** `Llama-3.2-3B-Instruct-Q5_K_M` with highest accuracy.

>  Full analysis: [docs/benchmark_report_llamacpp.md](../docs/benchmark_report_llamacpp.md)

---

## Model Selection Rationale

| Scenario | Model | Reason |
|----------|-------|--------|
| **Accuracy-first** (default) | Llama-3.2-3B Q5_K_M | 91.5% hit rate|
| **Speed-first** | Llama-3.2-3B Q4_K_M | 43% faster TTFT, 84% accuracy |
| Not recommended | Phi-4-mini (both) | 69% accuracy |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `llama-cpp-python` | GGUF inference engine |
| `sentence-transformers` | Multilingual embedding model |
| `numpy` | Embedding vector operations |
| `psutil` | CPU / RAM monitoring during benchmark |
| `matplotlib` | Benchmark chart generation |

---
