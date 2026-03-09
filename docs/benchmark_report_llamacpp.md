# Benchmark Evaluation Report (llama-cpp CPU Edition)

> CPU & Colab environment · 6 models × 10 queries · benchmark_cases.json

> Covers: Llama-3.2-3B Q5_K_M / Qwen2.5-3B Q5_K_M / Phi-4-mini Q5_K_M

---

## Evaluation Environment

| Item | Details |
|------|---------|
| Inference Engine | llama.cpp |
| Inference Device | CPU |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |
| Chunk Version | chunks.json (bilingual, 55 chunks) |
| Benchmark Cases | benchmark_cases.json (10 queries) |
| Metrics | Keyword Hit Rate / TTFT / TPS |

---

## 1. Overall Results

| Model | Hit Rate | TTFT (avg) | TPS | shared_spec | single_product | gpu_comparison |
|-------|:--------:|:----------:|:---:|:--------:|:-----------:|:--------------:|
| **Llama-3.2-3B Q5_K_M** | **91.5%** | 127,845 ms | 2.2 | 85.8% | 100% | **100%** |
| Llama-3.2-3B Q4_K_M | 84.0% | **73,022 ms** | **2.6** | 77.5% | 100% | 87.5% |
| Qwen2.5-3B Q5_K_M | 82.5% | 145,409 ms | 2.2 | 70.8% | 100% | 100% | **100%** |
| Qwen2.5-3B Q4_K_M | 77.0% | 81,512 ms | 2.6 | 74.2% | 75.0% | 87.5% |
| Phi-4-mini Q4_K_M | 69.0% | 96,480 ms | 2.2 | 73.3% | 75.0% | 50.0% |
| Phi-4-mini Q5_K_M | 69.0% | 163,269 ms | 1.9 | 65.0% | 75.0% | 75.0% |

---

## 2. Per-Query Hit Rate Breakdown

| Q# | Type | Query Summary | Llama Q4 | Llama Q5 | Qwen Q4 | Qwen Q5 | Phi Q4 | Phi Q5 |
|----|------|--------------|:--------:|:--------:|:-------:|:-------:|:------:|:------:|
| Q1 | shared_spec | Supported OS (ZH) | 100% | 100% | 100% | 100% | 100% | 100% |
| Q2 | single_product | BYH GDDR7 / AI Boost (ZH) | 100% | 100% | 100% | 100% | 100% | 100% |
| Q3 | shared_spec | Max RAM / SO-DIMM (ZH) | 75% | 75% | 75% | 75% | 50% | 50% |
| Q4 | shared_spec | Thunderbolt port count (ZH) | 100% | 100% | 100% | 100% | 100% | 100% |
| Q5 | gpu_comparison | BZH vs BXH TDP (ZH) | 75% | 100% | 75% | 100% | 75% | 100% |
| Q6 | shared_spec | Keyboard / Audio / Webcam (EN) | 40% | 40% | 20% | 0% | 40% | 40% |
| Q7 | shared_spec | Display panel / refresh rate (EN) | 50% | 100% | 50% | 50% | 50% | 50% |
| Q8 | shared_spec | Battery capacity (EN) | 100% | 100% | 100% | 100% | 100% | 50% |
| Q9 | single_product | BXH storage options (EN) | 100% | 100% | 50% | 100% | 50% | 50% |
| Q10 | gpu_comparison | BYH vs BXH gaming recommendation (EN) | 100% | 100% | 100% | 100% | 25% | 50% |

---

## 3. Observations & Analysis

### 3.1 Best Model: Llama-3.2-3B Q5_K_M (91.5%)

**91.5% Hit Rate** — the highest among all 6 models, and the only one without a complete zero-hit query:

- `single_product` (Q2, Q9): **100%** — precise single-model spec extraction fully correct
- `gpu_comparison` (Q5, Q10): **100%** — strongest cross-model comparison reasoning
- `shared_spec` (Q1–Q4, Q6–Q8): **85.8%** — only weakness is Q6 (multi-field combined English query)

---

### 3.2 Speed-Optimized Alternative: Llama-3.2-3B Q4_K_M (84.0%)

When the use case demands faster response times, Q4 is the best trade-off:

- TTFT **73,022 ms** — **43% faster** than Q5
- TPS **2.6 tok/s** — highest across all models
- Hit Rate 84.0% — only 7.5 percentage points below Q5

Suitable for deployment where conversational fluency is prioritized over marginal accuracy gains.

---

### 3.3 Qwen2.5-3B: Higher Quantization Hurts Accuracy

Qwen shows an **anomalous pattern**: Q5 accuracy (82.5%) is slightly higher than Q4 (77.0%) overall, but:

- Q5 `shared_spec` hit rate (70.8%) is **lower** than Q4 (74.2%)
- Q5 Q6 hit rate is **0%** — complete retrieval failure
- Q5 is 78% slower than Q4 (145,409 ms vs 81,512 ms)

Root cause analysis: 
Q6 asks about three separate spec fields (Keyboard + Audio + Webcam) in English. Qwen Q5 produces hallucinated responses claiming the data does not exist ("detailed information is not listed"), even though the bilingual chunks contain all required information. 
This indicates Qwen's weaker ability to synthesize multi-field information from context when queries are phrased in English.

---

### 3.4 Phi-4-mini: Lowest Accuracy, Not Suitable for This Use Case

Both quantization variants score only **69.0%**, with distinct failure modes:

- **Q10 gpu_comparison: only 25–50%** — the model incorrectly introduces BZH data when asked to compare BYH vs BXH (e.g., *"BZH has more memory (24GB) than both byh and bxh"*), drifting off-topic
- **Q9 storage: only 50%** — fabricates non-existent specs such as "up to four slots" (actual: 2 M.2 slots)
- Phi-4 is Microsoft's 4B parameter model; despite the larger size, TPS (1.9–2.2) is lower than 3B competitors

**Conclusion: Phi-4-mini is not recommended** for high-precision spec Q&A tasks.

---

### 3.5 Persistent Weakness: Q6 Multi-Field English Query

Q6 (*"What are the keyboard, audio, and webcam specs of the AORUS MASTER 16?"*) is the **common weak point across all models**:

| Model | Q6 Hit Rate | Root Cause |
|-------|:-----------:|-----------|
| Llama Q4/Q5 | 40% | Only Keyboard chunk retrieved; Audio/Webcam missed |
| Qwen Q4 | 20% | Only RGB keyword hit; model claims other data unavailable |
| Qwen Q5 | **0%** | Model claims all three fields are unavailable |
| Phi Q4/Q5 | 40% | Same as Llama; Audio/Webcam not retrieved |

**Root cause:** Although bilingual chunks have been created, `Keyboard`, `Audio`, and `Webcam` remain three **separate chunks**. With `top_k=5`, all three cannot be retrieved simultaneously. 

**Proposed fix:** Merge these three fields into a single "Input & Multimedia" composite chunk, or increase `top_k` to 8 for multi-field queries.

---

### 3.6 TTFT Pattern Analysis

All models show a **TTFT minimum at Q2** (15,000–35,000 ms): Q2 queries a specific model's GPU, which is directly matched by the key filter and returned without embedding search — validating the effectiveness of the key filter fast path design.

TTFT peaks occur at **Q1 (long context)** and **Q7 (Display field with many attributes)**, confirming that context length is the dominant factor in TTFT variance.

CPU utilization stays consistently at **80–100%** across all queries, indicating the system is fully CPU-bound.

---

## 4. Conclusions & Recommendations

### Deployment Recommendation

| Scenario | Recommended Model | Rationale |
|----------|------------------|-----------|
| **Accuracy-first** (default) | Llama-3.2-3B Q5_K_M | Hit Rate 91.5% |
| **Speed-first** | Llama-3.2-3B Q4_K_M | 43% faster TTFT, still 84% accuracy |

### Next Steps for Improvement

1. **Merge Keyboard/Audio/Webcam into one chunk** → Resolves Q6 incomplete retrieval.
2. **Add GPU inference benchmarks** → GPU expected to reduce TTFT to ~5–15 s
3. **Dynamic top_k** → Increase to top_k=8 for detected multi-field queries
4. **Upgrade to multilingual-e5-base** → Better recall for English multi-keyword queries vs MiniLM

---

## 5. Raw Results

All benchmark JSON files and PNG charts are available in the [`results/llama-cpp-cpu/`](../results/llama-cpp-cpu/) directory.
