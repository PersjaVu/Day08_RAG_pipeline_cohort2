"""
Retrieval-only Evaluation (KHÔNG cần LLM / Gemini).

Khi free tier Gemini hết quota, vẫn đo được CHẤT LƯỢNG RETRIEVAL thật vì
retrieval (Task 9) chạy hoàn toàn local (MiniLM + BM25 + ChromaDB).

Đo trên toàn bộ golden dataset, so sánh A/B 2 config:
    - hybrid_rerank : semantic + lexical + RRF rerank (Task 9 retrieve)
    - dense_only    : chỉ semantic search (Task 5)

Metrics (proxy, không cần judge LLM):
    - context_recall_proxy   : % từ khoá trong expected_context xuất hiện ở chunk lấy về
    - hit@k                  : có ít nhất 1 expected_context khớp tốt trong top-k không (0/1)
    - avg_semantic_score     : điểm cosine trung bình của chunk top-1 (chỉ config dense)

Chạy:
    python group_project/evaluation/retrieval_eval.py
"""

import io
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve

GOLDEN = Path(__file__).parent / "golden_dataset.json"
OUT_JSON = Path(__file__).parent / "retrieval_results.json"

TOP_K = 5
RECALL_TERM_THRESHOLD = 0.5  # 1 expected_context coi là "khớp" nếu >=50% từ khoá xuất hiện


def _norm(text: str) -> str:
    """Bỏ dấu + lowercase để so khớp tiếng Việt rộng rãi hơn."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _term_recall(expected_terms: list[str], joined_ctx: str) -> tuple[float, int]:
    """
    Với mỗi expected term, tính tỉ lệ từ khoá (>=4 ký tự) xuất hiện trong context.
    Trả về (recall trung bình các term, số term được coi là 'hit').
    """
    nctx = _norm(joined_ctx)
    recalls, hits = [], 0
    for term in expected_terms:
        words = [w for w in _norm(term).split() if len(w) >= 4]
        if not words:
            continue
        found = sum(1 for w in words if w in nctx)
        r = found / len(words)
        recalls.append(r)
        if r >= RECALL_TERM_THRESHOLD:
            hits += 1
    avg = sum(recalls) / len(recalls) if recalls else 0.0
    return avg, hits


def _retrieve(config: str, query: str) -> list[dict]:
    if config == "dense_only":
        return semantic_search(query, top_k=TOP_K)
    return retrieve(query, top_k=TOP_K)


def evaluate(config: str, dataset: list[dict]) -> dict:
    per_case, recalls, hit_flags, scores = [], [], [], []
    for item in dataset:
        q = item["question"]
        exp = item.get("expected_context", [])
        if isinstance(exp, str):
            exp = [exp]
        chunks = _retrieve(config, q)
        joined = " ".join(c.get("content", "") for c in chunks)
        recall, hits = _term_recall(exp, joined)
        hit = 1 if hits > 0 else 0
        top_score = chunks[0]["score"] if chunks else 0.0

        recalls.append(recall)
        hit_flags.append(hit)
        scores.append(top_score)
        per_case.append({
            "question": q,
            "recall_proxy": round(recall, 3),
            "hit@5": hit,
            "top_score": round(float(top_score), 4),
            "n_chunks": len(chunks),
        })

    n = len(dataset)
    return {
        "config": config,
        "context_recall_proxy": round(sum(recalls) / n, 3),
        "hit_rate@5": round(sum(hit_flags) / n, 3),
        "avg_top_score": round(sum(scores) / n, 4),
        "per_case": per_case,
    }


def main():
    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"Loaded {len(dataset)} questions")
    results = {}
    for cfg in ("hybrid_rerank", "dense_only"):
        print(f"\n=== {cfg} ===")
        r = evaluate(cfg, dataset)
        results[cfg] = r
        print(f"  context_recall_proxy = {r['context_recall_proxy']}")
        print(f"  hit_rate@5           = {r['hit_rate@5']}")
        print(f"  avg_top_score        = {r['avg_top_score']}")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
