"""
benchmark.py — AORUS MASTER 16 RAG 定量評測  (v3)
══════════════════════════════════════════════════
變更（相較 v2）：
  [1] 自動偵測推論裝置
      - llama.cpp 確實使用 GPU (n_gpu_layers > 0 且 nvidia-smi 偵測到用量)
        → 記錄 VRAM 指標
      - llama.cpp 強制 CPU (n_gpu_layers = 0 或偵測不到 GPU)
        → 記錄 RAM / CPU 指標
  [2] CPU 模式新增指標
      - Peak RAM Delta (MB)：推論期間 RAM 增量
      - Peak RAM Total (MB)：進程總 RSS
      - CPU Avg % / CPU Max %：推論期間 CPU 使用率
  [3] 圖表標題加入模型名稱
      "AORUS MASTER 16 RAG — Benchmark Results
       Model: Phi-3.1-mini-4k-instruct-Q4_K_M | Device: CPU"

定量指標：
  - Keyword Hit Rate（召回正確性）
  - TTFT (Time To First Token, 首字延遲)
  - TPS (Tokens Per Second, 生成速度)
  - [GPU] VRAM Used / Model VRAM Delta
  - [CPU] Peak RAM Delta / CPU Avg % / CPU Max %

執行：
    python benchmark.py
    python benchmark.py --cases data/benchmark_cases.json
    python benchmark.py --chunks data/chunks.json --emb data/embeddings.npy
"""

import json
import os
import argparse
import subprocess
import threading
import time
import psutil

from vector_index import VectorIndex
from retrieval_generate import (
    retrieve,
    build_context,
    generate_stream,
    load_llm,
    DEFAULT_MODEL_PATH,
)

DEFAULT_CASES_PATH = "data/benchmark_cases.json"


# ══════════════════════════════════════════════════════════
# DEVICE DETECTION
# ══════════════════════════════════════════════════════════

def get_vram_usage() -> dict:
    """嘗試讀取 nvidia-smi，失敗回傳空 dict。"""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 4:
            return {}
        return {
            "gpu_name": parts[0],
            "used_mb":  int(parts[1]),
            "total_mb": int(parts[2]),
            "free_mb":  int(parts[3]),
        }
    except Exception:
        return {}


def detect_device(llm, vram_before: dict, vram_after: dict) -> dict:
    """
    判斷 llama.cpp 實際使用的裝置。
    條件：nvidia-smi 可用 且 VRAM 用量增加 → GPU 模式
    否則 → CPU 模式
    """
    if vram_before and vram_after:
        delta = vram_after["used_mb"] - vram_before["used_mb"]
        if delta > 100:   # 至少增加 100MB 才算真正用到 GPU
            return {
                "device":         "GPU",
                "gpu_name":       vram_after["gpu_name"],
                "vram_total_mb":  vram_after["total_mb"],
                "vram_used_mb":   vram_after["used_mb"],
                "model_vram_mb":  delta,
                "within_4gb":     vram_after["used_mb"] <= 4096,
            }
    return {"device": "CPU"}


# ══════════════════════════════════════════════════════════
# CPU RESOURCE MONITOR
# ══════════════════════════════════════════════════════════

class CPUMonitor:
    """
    在背景執行緒取樣 CPU% 與 RSS，
    用法：
        with CPUMonitor() as mon:
            ... 推論 ...
        stats = mon.stats()
    """
    def __init__(self, interval: float = 0.1):
        self._interval  = interval
        self._process   = psutil.Process(os.getpid())
        self._cpu_samples: list[float] = []
        self._ram_before  = 0.0
        self._ram_peak    = 0.0
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            rss = self._process.memory_info().rss / 1024 ** 2
            if rss > self._ram_peak:
                self._ram_peak = rss
            time.sleep(self._interval)

    def __enter__(self):
        self._ram_before = self._process.memory_info().rss / 1024 ** 2
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)

    def stats(self) -> dict:
        samples = self._cpu_samples or [0.0]
        return {
            "ram_delta_mb": round(self._ram_peak - self._ram_before, 1),
            "ram_peak_mb":  round(self._ram_peak, 1),
            "cpu_avg_pct":  round(sum(samples) / len(samples), 1),
            "cpu_max_pct":  round(max(samples), 1),
        }


# ══════════════════════════════════════════════════════════
# CASES LOADER
# ══════════════════════════════════════════════════════════

def load_cases(path: str) -> list[tuple[str, list[str], str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Error] 找不到題目檔案：{path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for i, item in enumerate(raw):
        query    = item.get("query", "").strip()
        keywords = item.get("keywords", [])
        qtype    = item.get("type", "unknown")
        if not query:
            print(f"[Warning] 第 {i+1} 題 query 為空，跳過")
            continue
        cases.append((query, keywords, qtype))
    print(f"[✓] 題目載入完成：{len(cases)} 題 ← {path}")
    return cases


# ══════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════

def run_benchmark(
    index:      VectorIndex,
    llm,
    cases:      list[tuple[str, list[str], str]],
    device_info: dict,
    model_name: str,
    save_path:  str = "benchmark_results.json",
) -> None:
    is_gpu = device_info.get("device") == "GPU"

    print("\n" + "═" * 65)
    print("  RAG BENCHMARK")
    print("═" * 65)
    print(f"  模型     : {model_name}")
    print(f"  裝置     : {device_info.get('device', '?')}")

    if is_gpu:
        print(f"  GPU      : {device_info.get('gpu_name', '?')}")
        print(f"  模型佔用  : {device_info.get('model_vram_mb', '?')} MB VRAM")
        print(f"  總 VRAM  : {device_info.get('vram_used_mb', '?')} / "
              f"{device_info.get('vram_total_mb', '?')} MB")
        print(f"  4GB 符合 : {'✅ 是' if device_info.get('within_4gb') else '⚠️  超過 4GB'}")
    else:
        proc = psutil.Process(os.getpid())
        print(f"  RAM 使用  : {proc.memory_info().rss / 1024**2:.0f} MB (目前進程)")
        print(f"  CPU 核心  : {psutil.cpu_count(logical=True)} 顆邏輯核心")

    results = []

    for i, (query, expected_kw, qtype) in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] [{qtype}] {query}")
        print("─" * 55)

        retrieved = retrieve(index, query)
        context   = build_context(retrieved)

        answer_parts: list[str] = []
        gen_metrics:  dict      = {}

        print("回答：", end="", flush=True)

        # ── 依裝置決定是否啟動 CPU Monitor ──────────────
        if is_gpu:
            for token, m in generate_stream(llm, query, context):
                if token:
                    print(token, end="", flush=True)
                    answer_parts.append(token)
                if m:
                    gen_metrics = m
            resource_metrics: dict = {}
        else:
            with CPUMonitor() as monitor:
                for token, m in generate_stream(llm, query, context):
                    if token:
                        print(token, end="", flush=True)
                        answer_parts.append(token)
                    if m:
                        gen_metrics = m
            resource_metrics = monitor.stats()

        print()

        answer   = "".join(answer_parts)
        hit_kws  = [kw for kw in expected_kw if kw.lower() in answer.lower()]
        hit_rate = len(hit_kws) / len(expected_kw) if expected_kw else 0.0

        print(f"  ✔ Keywords : {hit_kws} / {expected_kw}  hit={hit_rate:.0%}")
        print(f"  ⏱ TTFT={gen_metrics.get('ttft_ms','?')} ms  "
              f"TPS={gen_metrics.get('tps','?')} tok/s  "
              f"Total={gen_metrics.get('total_ms','?')} ms  "
              f"Tokens={gen_metrics.get('total_tokens','?')}")

        if is_gpu:
            vram_now = get_vram_usage()
            if vram_now:
                print(f"  🖥 VRAM={vram_now.get('used_mb','?')} MB")
                resource_metrics = {"vram_used_mb": vram_now.get("used_mb")}
        else:
            print(f"  💾 RAM Δ={resource_metrics.get('ram_delta_mb','?')} MB  "
                  f"RAM Peak={resource_metrics.get('ram_peak_mb','?')} MB  "
                  f"CPU avg={resource_metrics.get('cpu_avg_pct','?')}%  "
                  f"CPU max={resource_metrics.get('cpu_max_pct','?')}%")

        results.append({
            "id":           i,
            "query":        query,
            "query_type":   qtype,
            "answer":       answer,
            "expected_kw":  expected_kw,
            "hit_keywords": hit_kws,
            "hit_rate":     hit_rate,
            **gen_metrics,
            **resource_metrics,
        })

    # ── Summary ──────────────────────────────────────────
    avg_hit  = sum(r["hit_rate"] for r in results) / len(results)
    avg_ttft = sum(r.get("ttft_ms", 0) for r in results) / len(results)
    avg_tps  = sum(r.get("tps", 0) for r in results) / len(results)

    print("\n" + "═" * 65)
    print("  BENCHMARK SUMMARY")
    print("═" * 65)
    print(f"  模型                  : {model_name}")
    print(f"  裝置                  : {device_info.get('device')}")
    print(f"  平均 Keyword Hit Rate : {avg_hit:.1%}")
    print(f"  平均 TTFT             : {avg_ttft:.0f} ms")
    print(f"  平均 TPS              : {avg_tps:.1f} tok/s")

    if not is_gpu:
        avg_ram = sum(r.get("ram_delta_mb", 0) for r in results) / len(results)
        avg_cpu = sum(r.get("cpu_avg_pct", 0) for r in results) / len(results)
        print(f"  平均 RAM Delta        : {avg_ram:.0f} MB")
        print(f"  平均 CPU 使用率       : {avg_cpu:.1f} %")

    by_type: dict[str, list[float]] = {}
    for r in results:
        by_type.setdefault(r["query_type"], []).append(r["hit_rate"])
    print()
    for qtype, rates in by_type.items():
        print(f"  [{qtype:18s}]  hit={sum(rates)/len(rates):.1%}  n={len(rates)}")
    print("═" * 65)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "model_name":    model_name,
                    "device":        device_info.get("device"),
                    "avg_hit_rate":  round(avg_hit, 4),
                    "avg_ttft_ms":   round(avg_ttft, 1),
                    "avg_tps":       round(avg_tps, 2),
                    **device_info,
                },
                "cases": results,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"[Benchmark] 結果已儲存 → {save_path}")

    chart_path = save_path.replace(".json", ".png")
    _save_charts(results, avg_hit, avg_ttft, avg_tps,
                 model_name, device_info, chart_path)


# ══════════════════════════════════════════════════════════
# CHART  — 加入模型名稱與裝置資訊於標題
# ══════════════════════════════════════════════════════════

def _save_charts(
    results:     list[dict],
    avg_hit:     float,
    avg_ttft:    float,
    avg_tps:     float,
    model_name:  str,
    device_info: dict,
    save_path:   str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Chart] matplotlib 未安裝，跳過圖表。pip install matplotlib")
        return

    is_gpu = device_info.get("device") == "GPU"

    TYPE_COLORS = {
        "single_product": "#4C9BE8",
        "gpu_comparison":  "#E8834C",
        "shared_spec":     "#5DBE7A",
    }

    labels    = [f"Q{r['id']}" for r in results]
    hit_rates = [r["hit_rate"] * 100 for r in results]
    ttfts     = [r.get("ttft_ms", 0) for r in results]
    colors    = [TYPE_COLORS.get(r["query_type"], "#aaaaaa") for r in results]

    # ── 決定第三個子圖要顯示 VRAM 還是 RAM/CPU ───────────
    has_resource = (
        is_gpu and any(r.get("vram_used_mb") for r in results)
    ) or (
        not is_gpu and any(r.get("ram_peak_mb") for r in results)
    )

    nrows = 3 if has_resource else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 4 * nrows))
    ax1, ax2  = axes[0], axes[1]
    ax3       = axes[2] if has_resource else None

    # ── 主標題：加入模型名稱與裝置 ───────────────────────
    device_label = device_info.get("device", "Unknown")
    if is_gpu:
        device_label += f" ({device_info.get('gpu_name', '')})"
    fig.suptitle(
        f"AORUS MASTER 16 RAG — Benchmark Results\n"
        f"Model: {model_name}  |  Device: {device_label}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    # ── 子圖 1：Keyword Hit Rate ──────────────────────────
    bars = ax1.bar(labels, hit_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(avg_hit * 100, color="#E84C4C", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("Keyword Hit Rate (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("Keyword Hit Rate per Query")
    for bar, val in zip(bars, hit_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    ax1.legend(handles=patches + [
        plt.Line2D([0], [0], color="#E84C4C", linestyle="--", label=f"avg {avg_hit:.1%}")
    ], fontsize=8, loc="lower right")

    # ── 子圖 2：TTFT ──────────────────────────────────────
    ax2.plot(labels, ttfts, marker="o", color="#4C9BE8",
             linewidth=1.8, markersize=6, label="TTFT (ms)")
    ax2.axhline(avg_ttft, color="#E84C4C", linestyle="--", linewidth=1.2,
                label=f"avg {avg_ttft:.0f} ms")
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_title(f"Time To First Token per Query   |   avg TPS = {avg_tps:.1f} tok/s")
    ax2.legend(fontsize=9)
    max_ttft = max(ttfts) if ttfts else 1
    for x, y in zip(labels, ttfts):
        ax2.text(x, y + max_ttft * 0.02, f"{y:.0f}", ha="center", va="bottom", fontsize=8)

    # ── 子圖 3：Resource（VRAM 或 RAM/CPU）────────────────
    if ax3 is not None:
        if is_gpu:
            vrams = [r.get("vram_used_mb", 0) for r in results]
            ax3.bar(labels, vrams, color="#9B59B6", edgecolor="white", linewidth=0.5)
            ax3.axhline(4096, color="#E84C4C", linestyle="--", linewidth=1.2,
                        label="4GB limit (4096 MB)")
            ax3.set_ylabel("VRAM Used (MB)")
            ax3.set_title("GPU VRAM Usage per Query")
            ax3.legend(fontsize=9)
            for bar, val in zip(ax3.patches, vrams):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                         f"{val}", ha="center", va="bottom", fontsize=8)
        else:
            # 雙 Y 軸：RAM（左）+ CPU%（右）
            ram_peaks = [r.get("ram_peak_mb", 0) for r in results]
            cpu_avgs  = [r.get("cpu_avg_pct", 0) for r in results]

            color_ram = "#9B59B6"
            color_cpu = "#E8834C"

            bars3 = ax3.bar(labels, ram_peaks, color=color_ram,
                            edgecolor="white", linewidth=0.5, label="RAM Peak (MB)")
            ax3.set_ylabel("RAM Peak (MB)", color=color_ram)
            ax3.tick_params(axis="y", labelcolor=color_ram)
            ax3.set_title("CPU Mode: RAM Usage & CPU Utilization per Query")

            ax3b = ax3.twinx()
            ax3b.plot(labels, cpu_avgs, marker="s", color=color_cpu,
                      linewidth=1.8, markersize=6, label="CPU Avg %")
            ax3b.set_ylabel("CPU Avg (%)", color=color_cpu)
            ax3b.tick_params(axis="y", labelcolor=color_cpu)
            ax3b.set_ylim(0, 110)

            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

            for bar, val in zip(bars3, ram_peaks):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                         f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Benchmark] 圖表已儲存 → {save_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument("--chunks", default="data/chunks.json",       help="Chunks JSON 路徑")
    parser.add_argument("--emb",    default="data/embeddings.npy",    help="Embeddings 快取路徑")
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH,       help="GGUF 模型路徑")
    parser.add_argument("--cases",  default=DEFAULT_CASES_PATH,       help="題目 JSON 檔案路徑")
    parser.add_argument("--out",    default="benchmark_results.json", help="結果輸出路徑")
    args = parser.parse_args()

    # ── 載入 chunks ───────────────────────────────────────
    if not os.path.exists(args.chunks):
        print(f"[Error] 找不到 {args.chunks}，請先執行 chunk_create.py")
        exit(1)
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    # ── 載入題目 ──────────────────────────────────────────
    cases = load_cases(args.cases)

    # ── 建立 index ────────────────────────────────────────
    index = VectorIndex()
    index.build(chunks_data, emb_cache=args.emb)

    # ── VRAM 量測（load_llm 前後，才是正確的模型佔用量）──
    vram_before = get_vram_usage()
    llm         = load_llm(args.model)
    vram_after  = get_vram_usage()

    # ── 模型名稱：從路徑抽取檔名（去副檔名）─────────────
    model_name = os.path.splitext(os.path.basename(args.model))[0]

    # ── 自動判斷裝置 ──────────────────────────────────────
    device_info = detect_device(llm, vram_before, vram_after)
    print(f"\n[Device] 偵測到推論裝置：{device_info.get('device')}")
    if device_info.get("device") == "GPU":
        print(f"[Device] GPU: {device_info.get('gpu_name')}  "
              f"VRAM: {device_info.get('vram_used_mb')} MB")
    else:
        print(f"[Device] CPU 模式，將記錄 RAM / CPU 指標")

    # ── 執行 benchmark ────────────────────────────────────
    run_benchmark(
        index, llm, cases=cases,
        device_info=device_info,
        model_name=model_name,
        save_path=args.out,
    )

    # ── 清理 LLM ──────────────────────────────────────────
    if hasattr(llm, "close"):
        llm.close()
    if hasattr(llm, "_ctx"):
        llm._ctx = None