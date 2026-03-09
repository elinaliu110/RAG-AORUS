# AORUS MASTER 16 — AI Spec Assistant (vLLM GPU Edition)

This project implements a RAG-based spec Q&A system using **vLLM + AWQ INT4 quantized models** on GPU (Google Colab Tesla T4).

---

## Project Structure

```
aorus-rag/
├── README.md               
├── pyproject.toml                  # vLLM dependencies
│
├── src/
│   ├── chunk_create.py             # specs.csv → bilingual chunks.json
│   ├── vector_index.py             # Vector index (build / search / exact lookup)
│   ├── retrieval_generate.py       # RAG core: filter → retrieve → generate
│   └── benchmark.py                # Benchmark runner 
│
├── data/
│   ├── specs.csv                   # Raw spec data
│   ├── chunks.json                 # Bilingual chunks (55 chunks)
│   ├── benchmark_cases.json        # Evaluation queries (10 cases)
│   └── embeddings.npy              # Vector cache
│
├── results/vllm-gpu                # Benchmark output
└── docs/
    └── benchmark_report.md
```

---

## Test Environment

| Item | Details |
|------|---------|
| Inference Engine | vLLM (AWQ INT4 quantization) |
| GPU | Tesla T4 (Google Colab) |
| Total VRAM | 15,360 MB (idle: 567 MB)|
| 4 GB Constraint Criterion | Model VRAM Δ ≤ 4,096 MB |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |
| Chunk Version | chunks.json (bilingual, 55 chunks) |
| Benchmark Cases | benchmark_cases.json (10 queries) |

---

## Requirements

| Item | Requirement |
|------|-------------|
| Python | ≥ 3.11 |
| GPU | NVIDIA GPU (VRAM ≥ 4 GB; Colab T4 works) |
| CUDA | 12.x (required by vLLM) |
| Package manager | `uv` |

> **Note: vLLM only supports Linux + NVIDIA GPU.** It cannot run on local Windows/macOS. Please use Google Colab or a Linux GPU server.

---

## Setup (Colab)

### 1. Enable GPU Runtime

Colab menu → **Runtime** → **Change runtime type** → select **T4 GPU** (or higher).

```bash
# Verify GPU is available
!nvidia-smi
```

### 2. HuggingFace Token & Model Access

Some models (Llama family) are **gated**, you must:

1. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Visit each model page and accept the **User Agreement**:
   - [AMead10/Llama-3.2-3B-Instruct-AWQ](https://huggingface.co/AMead10/Llama-3.2-3B-Instruct-AWQ) — requires accepting [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) license first
   - [AMead10/Llama-3.2-1B-Instruct-AWQ](https://huggingface.co/AMead10/Llama-3.2-1B-Instruct-AWQ)
   - [casperhansen/llama-3.2-3b-instruct-awq](https://huggingface.co/casperhansen/llama-3.2-3b-instruct-awq)
   - Qwen models are open-access — no agreement required

3. Set your token in Colab (pick one method):

```python
# Option A: Login via huggingface_hub (recommended — token won't appear in output)
from huggingface_hub import login
login()  # Opens an input prompt; paste your token there

# Option B: Environment variable
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Option C: Colab Secrets (most secure)
# Left sidebar → Secrets → Add HF_TOKEN
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

### 3. Clone the Repository

```bash
!git clone https://github.com/elinaliu110/aorus-rag.git
%cd aorus-rag
```

### 4. Install Dependencies

**Option A: `uv` (recommended — fastest)**

```bash
# Install uv
!pip install uv -q

# Install all project dependencies from pyproject.toml
!uv sync
```

> `pyproject.toml` lists all dependencies. For local Linux environments.

---

## Quick Start

### Step 0: Create Chunks (if not yet done)

```bash
!uv run python src/chunk_create.py \
    --input  data/specs.csv \
    --output data/chunks.json
```

### Step 1: Build Vector Index

```bash
!uv run python src/vector_index.py \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy
```

### Step 2: Run Benchmark - Simulate 4 GB VRAM Constraint

The T4 has 15 GB VRAM. Use `--gpu-util` to cap allocation and simulate a consumer 4 GB GPU:

```bash
# T4 requires --enforce-eager (no FlashAttention2 support on compute capability 7.5)
!uv run python src/benchmark.py \
    --models AMead10/Llama-3.2-3B-Instruct-AWQ \
    --cases  data/benchmark_cases.json \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy \
    --max-model-len 1024 \
    --gpu-util 0.19 \
    --out-dir results \
    --enforce-eager
```

### Step 3: Run Benchmark — Multiple Models

```bash
# T4 requires --enforce-eager (no FlashAttention2 support on compute capability 7.5)
!uv run python src/benchmark.py \
    --models AMead10/Llama-3.2-3B-Instruct-AWQ,casperhansen/llama-3.2-3b-instruct-awq,Qwen/Qwen2.5-3B-Instruct-AWQ \
    --cases  data/benchmark_cases.json \
    --chunks data/chunks.json \
    --emb    data/embeddings.npy \
    --max-model-len 1024 \
    --gpu-util 0.19 \
    --out-dir results \
    --enforce-eager
```

Output files in `results/`:
- `benchmark_{model_name}.json` — per-query detailed results
- `benchmark_{model_name}.png` — per-model performance charts
- `benchmark_comparison.json` — aggregated multi-model summary
- `benchmark_comparison.png` — comparison chart


## Models Tested

| Model Family | Params | Quant | HuggingFace ID | Access |
|-------------|:------:|:-----:|----------------|:------:|
| Llama-3.2-Instruct (AMead10) | 3B | AWQ INT4 | `AMead10/Llama-3.2-3B-Instruct-AWQ` | Gated |
| Llama-3.2-Instruct (casper) | 3B | AWQ INT4 | `casperhansen/llama-3.2-3b-instruct-awq` | Gated |
| Qwen2.5-Instruct | 3B | AWQ INT4 | `Qwen/Qwen2.5-3B-Instruct-AWQ` | Open |
| Llama-3.2-Instruct (AMead10) | 1B | AWQ INT4 | `AMead10/Llama-3.2-1B-Instruct-AWQ` | Gated |
| Llama-3.2-Instruct (casper) | 1B | AWQ INT4 | `casperhansen/llama-3.2-1b-instruct-awq` | Gated |
| Qwen2.5-Instruct | 1.5B | AWQ INT4 | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | Open |

---

## Benchmark Summary

### 3B Models Results

| Model | Hit Rate | Avg TTFT | Avg TPS | Model VRAM Δ | Inference VRAM Δ | 4GB Ready |
|-------|:--------:|:--------:|:-------:|:------------:|:------------:|:---------:|
| **AMead10 Llama-3.2-3B** | **84.0%** | 269 ms | 36.1 tok/s | 3,492 MB | 4,059 MB | ✅ |
| casperhansen Llama-3.2-3B | 81.5% | **241 ms** | **37.9 tok/s** | 3,540 MB | 4,107 MB | ❌ +11 MB |
| Qwen2.5-3B | 58.2% | 274 ms | 31.0 tok/s | **3,468 MB** | 4,035 MB | ✅ |

### 1B Models Results

| Model | Hit Rate | Avg TTFT | Avg TPS | Model VRAM Δ | Inference VRAM Δ | 4GB Ready |
|-------|:--------:|:--------:|:-------:|:------------:|:---------:|
| Qwen2.5-1.5B | 53.7% | 202 ms | 36.5 tok/s | 3,462 MB | 4,015 MB | ✅ |
| casperhansen Llama-3.2-1B | 40.3% | **132 ms** | 56.6 tok/s | **3,398 MB** | 3,965 MB | ✅ |
| AMead10 Llama-3.2-1B | 32.0% | 133 ms | **57.4 tok/s** | 3,446 MB | 4,032 MB | ✅ |

> **Recommended: `AMead10/Llama-3.2-3B-Instruct-AWQ`** — highest accuracy (84%), strictly 4 GB compliant.

> **1B models are not recommended:** Overall accuracy of 32–40% is not acceptable.

> Full per-query analysis → [docs/benchmark_report.md](docs/benchmark_report.md)

---


## Script Reference

### `benchmark.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | `your-model-name` | Model ID(s), comma-separated for multiple |
| `--base-url` | `http://localhost:8000/v1` | vLLM server address |
| `--cases` | `data/benchmark_cases.json` | Benchmark cases JSON path |
| `--chunks` | `data/chunks.json` | Chunks JSON path |
| `--emb` | `data/embeddings.npy` | Embedding cache path |
| `--out-dir` | `results` | Output directory |
| `--gpu-util` | `0.85` | vLLM VRAM utilization fraction (0–1) |
| `--max-model-len` | `4096` | Context window size in tokens |
| `--enforce-eager` | `False` | **Required for T4** — disables FlashAttention2 |

### `retrieval_generate.py`

RAG core module with four stages:

| Stage | Function | Description |
|-------|----------|-------------|
| C-1 | `extract_product_filter` | Identifies single-product queries (BZH / BYH / BXH) vs. comparison queries |
| C-1 | `extract_key_filter` | Maps query keywords to canonical spec sections (GPU, Battery, Display, etc.) |
| C-2 | `retrieve` | Key filter → exact lookup; else → vector similarity (top_k × 3 for cross-SKU coverage) |
| C-3 | `load_llm` | Creates OpenAI-compatible client connected to the running vLLM server |
| C-4 | `generate_stream` | Streaming generation with TTFT / TPS / total_tokens metrics |

### `vector_index.py`

Supports three multilingual embedding models:

| Alias | Model | Notes |
|-------|-------|-------|
| `minilm` (default) | `paraphrase-multilingual-MiniLM-L12-v2` | Lightweight, fast, recommended |
| `e5-base` | `intfloat/multilingual-e5-base` | Higher accuracy |
| `e5-large` | `intfloat/multilingual-e5-large` | Highest accuracy, slower |

To rebuild the cache with a different model:

```bash
!uv run python src/vector_index.py --model e5-base --force
```

---

## 4 GB VRAM Constraint

The test GPU is a Tesla T4 (15 GB), but the evaluation criterion follows the interview requirement: **consumer-grade 4 GB VRAM**.

Assessment method: Inference VRAM after loading ≤ 4,096 MB.

| Model | Params | Model VRAM Δ | Inference VRAM | 4GB Ready |
|-------|:------:|:------------:|:--------------:|:---------:|
| AMead10 Llama 3B | 3B | 3,492 MB | 4,059 MB | ✅ |
| casperhansen Llama 3B | 3B | 3,540 MB | 4,107 MB | ❌ +11 MB |
| Qwen2.5 3B | 3B | 3,468 MB | 4,035 MB | ✅ |
| AMead10 Llama 1B | 1B | 3,446 MB | 4,015 MB | ✅ |
| casperhansen Llama 1B | 1B | 3,398 MB | 3,965 MB | ✅ |
| Qwen2.5 1.5B | 1.5B | 3,462 MB | 4,032 MB | ✅ |

T4-specific notes:
- **`--enforce-eager` is required**: T4 compute capability is 7.5; FlashAttention2 requires ≥ 8.0 (A100/H100). The script automatically sets `VLLM_ATTENTION_BACKEND=XFORMERS`.
- Default `--gpu-util 0.85` on a T4 allocates ~13 GB VRAM.
- To simulate 4 GB: use `--gpu-util 0.19` (VRAM idle: 567 MB + 15 GB × 0.19 ≈ 4 GB).

---

## vLLM GPU vs llama.cpp CPU

| Metric | vLLM GPU (AMead10 3B) | llama.cpp CPU (Q5_K_M) |
|--------|:---------------------:|:----------------------:|
| Hit Rate | 84.0% | **91.5%** |
| Avg TTFT | **269 ms** | 127,845 ms |
| Avg TPS | **36.1 tok/s** | 2.2 tok/s |

- GPU TTFT is ~**475× faster**; TPS is ~**16× faster**
- Full root-cause analysis in the [benchmark report](docs/benchmark_report.md)

---

## Results & Docs

- Full benchmark analysis (6 models × 10 queries): [docs/benchmark_report.md](docs/benchmark_report.md)
- Raw results JSON / PNG: [results/vllm-gpu/](results/vllm-gpu/)
- llama.cpp CPU edition: [README_llamacpp.md](README_llamacpp.md)
