"""
retrieval_generate.py — AORUS MASTER 16 RAG 核心函式庫
═══════════════════════════════════════════════════════
純函式定義，不直接執行。供以下模組 import 使用：
  - benchmark.py   → 定量評測
  - run_main.py    → 互動式問答入口

包含：
  - Stage C-1: extract_product_filter / extract_key_filter
  - Stage C-2: retrieve / build_context
  - Stage C-3: load_llm
  - Stage C-4: generate_stream（streaming + TTFT/TPS 測量）
"""

import json
import time
from typing import Generator

from vector_index import VectorIndex

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

DEFAULT_MODEL_PATH = "models/your-model.gguf"   # ← 請改成你的 GGUF 路徑

PRODUCTS = {
    "BZH": "AORUS MASTER 16 BZH",
    "BYH": "AORUS MASTER 16 BYH",
    "BXH": "AORUS MASTER 16 BXH",
}

# 比較型查詢關鍵字 → 不套 product filter
COMPARISON_KEYWORDS = {
    "比較", "差異", "差別", "哪個", "哪款", "推薦", "建議",
    "vs", "versus", "compare", "difference", "better", "best",
    "which", "recommend", "between", "should i", "choose",
}

# query 關鍵字 → chunk key 對應表
# 命中時直接 by key 精確取回，跳過 vector search
KEY_ALIASES: dict[str, str] = {
    "gpu": "Video Graphics", "顯卡": "Video Graphics", "顯示卡": "Video Graphics",
    "vram": "Video Graphics", "顯示記憶體": "Video Graphics",
    "cpu": "CPU", "處理器": "CPU", "晶片": "CPU",
    "螢幕": "Display", "display": "Display", "oled": "Display",
    "解析度": "Display", "dolby vision": "Display", "g-sync": "Display",
    "gsync": "Display", "hdr": "Display", "hz": "Display",
    "記憶體": "System Memory", "ram": "System Memory", "ddr": "System Memory",
    "儲存": "Storage", "ssd": "Storage", "m.2": "Storage", "nvme": "Storage",
    "電池": "Battery", "battery": "Battery", "續航": "Battery",
    "重量": "Weight", "weight": "Weight", "幾公斤": "Weight",
    "尺寸": "Dimensions (W x D x H)", "dimensions": "Dimensions (W x D x H)",
    "連接埠": "I/O Port", "port": "I/O Port", "usb": "I/O Port",
    "thunderbolt": "I/O Port", "hdmi": "I/O Port", "type-c": "I/O Port",
    "鍵盤": "Keyboard Type", "keyboard": "Keyboard Type",
    "音訊": "Audio", "speaker": "Audio", "喇叭": "Audio",
    "wifi": "Communications", "藍牙": "Communications", "bluetooth": "Communications",
    "wireless": "Communications", "connectivity": "Communications",
    "lan": "Communications", "network": "Communications", "無線": "Communications",
    "攝影機": "Webcam", "webcam": "Webcam", "camera": "Webcam",
    "os": "OS", "作業系統": "OS", "windows": "OS",
    "安全": "Security", "tpm": "Security",
    "變壓器": "Adapter", "adapter": "Adapter", "充電": "Adapter",
    "顏色": "Color", "color": "Color",
}


# ══════════════════════════════════════════════════════════
# STAGE C-1 — FILTER EXTRACTION
# ══════════════════════════════════════════════════════════

def extract_product_filter(query: str) -> str | None:
    """
    從 query 萃取 product filter。
    以下兩種情況回傳 None（不 filter）：
      - 偵測到比較型關鍵字（e.g. 哪個、推薦、vs）
      - query 同時提及兩個以上機型
    """
    q_upper = query.upper()
    q_lower = query.lower()

    if any(kw in q_lower for kw in COMPARISON_KEYWORDS):
        return None

    matched = [sid for sid in PRODUCTS if sid in q_upper]
    if len(matched) > 1:
        return None

    return matched[0] if matched else None


def extract_key_filter(query: str) -> str | None:
    """
    從 query 萃取 spec key filter。
    命中時可直接 by key 精確取回，跳過 vector search。
    """
    q_lower = query.lower()
    for alias, key in KEY_ALIASES.items():
        if alias in q_lower:
            return key
    return None


# ══════════════════════════════════════════════════════════
# STAGE C-2 — RETRIEVAL
# ══════════════════════════════════════════════════════════

def retrieve(index: VectorIndex, query: str, top_k: int = 5) -> list[dict]:
    """
    優化版檢索，依序執行：
      1. 萃取 product_filter、key_filter
      2. key_filter 命中 → 直接精確取回，跳過 vector search（省 encode 時間）
      3. vector search + metadata filter
      4. 無 product_filter 時補入 ALL summary chunk
      5. 去重後回傳
    """
    product_filter = extract_product_filter(query)
    key_filter     = extract_key_filter(query)

    # ── key filter 精確命中路徑 ──────────────────────────
    if key_filter:
        exact = index.get_by_key(key_filter, short_id=product_filter)
        if exact:
            # 無 product_filter（通用問題）補入 ALL summary
            supplement = index.get_by_short_id("ALL") if product_filter is None else []
            return _merge_unique(exact, supplement, top_k)

    # ── vector search 路徑 ───────────────────────────────
    # 無 product_filter 時 top_k × 3，確保三台相同規格的 chunk 都能進來
    search_k = top_k if product_filter else top_k * 3
    results  = index.search(query, top_k=search_k, product_filter=product_filter)

    # 無 product_filter 時補入 ALL summary
    if product_filter is None:
        all_summary = index.get_by_short_id("ALL")
        results     = _merge_unique(results, all_summary, search_k)

    return _dedup(results)[:top_k]


def build_context(chunks: list[dict], max_tokens: int = 1400) -> str:
    """
    組合 context，加入 token 預算控制避免 prompt 過長影響 TTFT。
    雙語 chunk 的 text 欄位為「中文 / 英文」合併，長度約為純中文的 2 倍，
    因此 max_tokens 預設值從 800 調整為 1400。
    粗估：中英混合約 1 字 = 0.6 token；英文約 1 字 = 1.3 token。
    保守用 0.7 係數覆蓋雙語混合情境。
    """
    lines, total = [], 0
    for c in chunks:
        est = int(len(c["text"]) * 0.7)
        if total + est > max_tokens:
            break
        lines.append(c["text"])
        total += est
    return "\n".join(lines)


def _dedup(chunks: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    return unique


def _merge_unique(
    primary:   list[dict],
    secondary: list[dict],
    limit:     int,
) -> list[dict]:
    return _dedup(primary + secondary)[:limit]


# ══════════════════════════════════════════════════════════
# STAGE C-3 — LLM 載入
# ══════════════════════════════════════════════════════════

def load_llm(model_path: str = DEFAULT_MODEL_PATH):
    from llama_cpp import Llama
    print(f"[LLM] 載入 llama.cpp 模型: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("[LLM] 模型載入完成。")
    return llm


# ══════════════════════════════════════════════════════════
# STAGE C-4 — STREAMING GENERATION + TTFT/TPS 測量
# ══════════════════════════════════════════════════════════

def generate_stream(
    llm,
    query:          str,
    context:        str,
    max_new_tokens: int = 200,
) -> Generator[tuple[str, dict], None, None]:
    """
    Yields (token_text, metrics_dict).
    metrics_dict 只在最後一次 yield 時有值：
      {"ttft_ms": float, "tps": float, "total_tokens": int, "total_ms": float}
    """
    prompt = (
        "<|system|>\n"
        "你是一位 GIGABYTE AORUS MASTER 16 AM6H 的專業規格專家。請根據以下提供的規格資料回答問題。"
        "若問題以中文提問請用繁體中文回答；若以英文提問請用英文回答。"
        "只根據資料內容回答，若資料中找不到答案請明確說明。\n<|end|>\n"
        f"<|user|>\n規格資料:\n{context}\n\n問題: {query}<|end|>\n"
        "<|assistant|>\n"
    )

    t_start      = time.perf_counter()
    t_first      = None
    total_tokens = 0

    stream = llm(
        prompt,
        max_tokens=max_new_tokens,
        stream=True,
        temperature=0.0,
        repeat_penalty=1.3,
        stop=["<|end|>", "<|user|>", "<|system|>", "\n\n\n"],  # 遇到這些立刻停
    )

    for chunk in stream:
        token_text = chunk["choices"][0]["text"]
        if not token_text:
            continue
        if t_first is None:
            t_first = time.perf_counter()
        total_tokens += 1
        yield token_text, {}

    t_end    = time.perf_counter()
    ttft_ms  = (t_first - t_start) * 1000 if t_first else 0.0
    total_ms = (t_end - t_start) * 1000
    gen_ms   = (t_end - t_first) * 1000 if t_first else 1.0
    tps      = total_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0

    yield "", {
        "ttft_ms":      round(ttft_ms, 1),
        "tps":          round(tps, 2),
        "total_tokens": total_tokens,
        "total_ms":     round(total_ms, 1),
    }