"""
vector_index_v2.py — AORUS MASTER 16 向量索引
═══════════════════════════════════════════════
修改重點（相較於 v1）：
  [1] search() 拿掉 "ALL" 強制保留，ALL summary 改由 retrieval.py 視情況帶入
  [2] 新增 get_by_key()：by spec key 精確取回，供 retrieval key filter 使用
  [3] 新增 get_by_short_id()：取出所有屬於該 short_id 的 chunk
  [4] __main__ 支援 CLI 參數，可指定 chunks、輸出 npy 路徑、embedding model
      方便比較不同 embedding model 的效果

支援的雙語 Embedding Model（中英文）：
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  （預設，輕量快速）
  - intfloat/multilingual-e5-base                                 （精度較高）
  - intfloat/multilingual-e5-large                                （精度最高，較慢）

執行：
    # 使用預設模型
    python vector_index_v2.py

    # 指定模型與路徑
    python vector_index_v2.py \\
        --chunks data/chunks.json \\
        --emb    data/embeddings_e5base.npy \\
        --model  intfloat/multilingual-e5-base

    # 比較三個模型（分別產出不同 npy）
    python vector_index_v2.py --model intfloat/multilingual-e5-base    --emb data/emb_e5base.npy
    python vector_index_v2.py --model intfloat/multilingual-e5-large   --emb data/emb_e5large.npy
    python vector_index_v2.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --emb data/emb_minilm.npy
"""

import json
import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

# 可用的雙語 embedding model 清單（供參考）
SUPPORTED_MODELS = {
    "minilm":   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5-base":  "intfloat/multilingual-e5-base",
    "e5-large": "intfloat/multilingual-e5-large",
}

DEFAULT_MODEL = SUPPORTED_MODELS["minilm"]


class VectorIndex:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        print(f"[Index] 載入 embedding 模型: {model_name}")
        self.model_name = model_name
        self.model      = SentenceTransformer(model_name)
        self.chunks:     list[dict]        = []
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[dict], emb_cache: str = "data/embeddings.npy") -> None:
        self.chunks = chunks
        if os.path.exists(emb_cache):
            print(f"[Index] Embedding 快取存在，跳過編碼 → {emb_cache}")
            self.embeddings = np.load(emb_cache)
            return
        print(f"[Index] 編碼 {len(chunks)} 個 chunks（model: {self.model_name}）...")
        # 使用雙語合併的 text 欄位（中文 / 英文），讓 embedding 同時涵蓋兩種語言語意
        texts = [c["text"] for c in chunks]

        # multilingual-e5 系列需要加 query/passage prefix 才能發揮最佳效果
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]

        embs  = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        self.embeddings = embs / np.maximum(norms, 1e-9)
        os.makedirs(os.path.dirname(emb_cache) or ".", exist_ok=True)
        np.save(emb_cache, self.embeddings)
        print(f"[Index] Embeddings 已儲存 → {emb_cache}")

    def search(
        self,
        query:          str,
        top_k:          int        = 5,
        product_filter: str | None = None,
    ) -> list[dict]:
        # multilingual-e5 系列 query 也需要加 prefix
        q_text = f"query: {query}" if "e5" in self.model_name.lower() else query
        q_emb  = self.model.encode([q_text])
        q_emb /= np.maximum(np.linalg.norm(q_emb), 1e-9)
        scores = (self.embeddings @ q_emb.T)[:, 0].copy()

        if product_filter:
            for i, c in enumerate(self.chunks):
                if c.get("short_id") != product_filter:
                    scores[i] = -1.0

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{**self.chunks[i], "_score": float(scores[i])} for i in top_idx]

    def get_by_key(
        self,
        key:      str,
        short_id: str | None = None,
    ) -> list[dict]:
        """by spec key 精確取回，可選擇限定 short_id。"""
        return [
            c for c in self.chunks
            if c.get("key") == key
            and (short_id is None or c.get("short_id") == short_id)
        ]

    def get_by_short_id(self, short_id: str) -> list[dict]:
        """取出所有屬於該 short_id 的 chunk。"""
        return [c for c in self.chunks if c.get("short_id") == short_id]


# ══════════════════════════════════════════════════════════
# MAIN — CLI 支援指定 model / chunks / emb 路徑
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="建立向量索引並測試搜尋")
    parser.add_argument(
        "--chunks", default="data/chunks.json",
        help="Chunks JSON 路徑（預設：data/chunks.json）",
    )
    parser.add_argument(
        "--emb", default="data/embeddings.npy",
        help="輸出 npy 路徑（預設：data/embeddings.npy）",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            "Embedding model 名稱，支援簡稱或完整 HuggingFace model ID\n"
            f"  簡稱對應：{SUPPORTED_MODELS}\n"
            f"  預設：{DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重新編碼，忽略既有快取",
    )
    args = parser.parse_args()

    # 支援簡稱（e.g. --model e5-base）
    model_name = SUPPORTED_MODELS.get(args.model, args.model)

    # 若 --force，刪除既有快取
    if args.force and os.path.exists(args.emb):
        os.remove(args.emb)
        print(f"[Index] 強制重建，已刪除快取 → {args.emb}")

    if not os.path.exists(args.chunks):
        print(f"[Error] 找不到 {args.chunks}，請先執行 chunk_create.py")
        exit(1)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    print(f"[Main] 成功讀取 {len(chunks_data)} 個 chunks ← {args.chunks}")

    # 雙語 chunks 偵測提示
    if any("text_zh" in c for c in chunks_data):
        print("[Info] 偵測到雙語 chunks（含 text_zh / text_en 欄位），embedding 將使用合併雙語 text。")
        print("[Info] ⚠️  如沿用舊 .npy 快取可能導致語意不對齊，建議搭配 --force 重新編碼。")

    # 建立索引
    index = VectorIndex(model_name=model_name)
    index.build(chunks_data, emb_cache=args.emb)

    # 測試搜尋（含英文無線查詢，驗證雙語 chunk 效果）
    test_queries = [
        "AORUS MASTER 16 BZH 的 GPU 是什麼型號？",
        "AORUS MASTER 16 BXH 支援最大多少 GB RAM？",
        "What AORUS MASTER 16 BYH Adapter？",
        "電池容量多少？",
        "What wireless connectivity does the AORUS MASTER 16 support?",  # 原本 Q6 失敗的案例
    ]
    print("\n" + "═" * 50)
    print(f"  搜尋測試  |  model: {model_name}")
    print("═" * 50)
    for q in test_queries:
        results = index.search(q, top_k=3)
        print(f"\n查詢: {q}")
        for i, res in enumerate(results):
            print(f"  {i+1}. [{res['short_id']:4s}] score={res['_score']:.4f}  {res['text'][:80]}...")
    print("═" * 50)