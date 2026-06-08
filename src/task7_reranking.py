"""
Task 7 — Reranking Module.

Ba phương pháp được implement đầy đủ:

1. RRF (Reciprocal Rank Fusion) — mặc định, không cần API
   Gộp nhiều ranked lists: RRF(d) = Σ 1/(k + rank_r(d))
   k=60 từ paper Cormack et al. (2009), giảm ảnh hưởng rank cao bất thường.

2. Cross-Encoder (Jina Reranker v2 API)
   Xét cặp (query, document) cùng lúc → chính xác hơn bi-encoder.
   Yêu cầu JINA_API_KEY trong .env.
   Nếu không có key → fallback sort theo score gốc.

3. MMR (Maximal Marginal Relevance)
   Cân bằng relevance vs diversity:
   MMR = λ * sim(q, d) - (1-λ) * max(sim(d, selected))
   Hữu ích khi muốn tránh trả về quá nhiều chunks trùng lặp.
"""

import os
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    k=60: hằng số làm mịn từ Cormack et al. 2009, giảm ưu tiên cực đoan cho rank 1.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    return results


# =============================================================================
# Cross-Encoder — Jina Reranker v2 API
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker v2 (multilingual).

    Nếu JINA_API_KEY không được set → fallback: sort theo score gốc.
    """
    if not JINA_API_KEY:
        print("  [reranker] JINA_API_KEY not set — returning candidates by original score")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    reranked = response.json()["results"]

    return [
        {**candidates[r["index"]], "score": r["relevance_score"]}
        for r in reranked
    ]


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — relevance vs diversity.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    lambda_param=0.7: ưu tiên relevance hơn diversity (70/30).
    Candidates phải có key 'embedding': list[float].
    """
    if not candidates:
        return []

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            if emb is None:
                continue
            relevance = _cosine_sim(query_embedding, emb)

            if selected_indices:
                max_sim = max(
                    _cosine_sim(emb, candidates[s]["embedding"])
                    for s in selected_indices
                    if candidates[s].get("embedding")
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [
        {**candidates[i], "score": _cosine_sim(query_embedding, candidates[i]["embedding"])}
        for i in selected_indices
        if candidates[i].get("embedding")
    ]


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.

    method="rrf"           → Reciprocal Rank Fusion (no API, default)
    method="cross_encoder" → Jina API (needs JINA_API_KEY, fallback to score sort)
    method="mmr"           → MMR (candidates must have 'embedding' key)
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    elif method == "mmr":
        raise ValueError("MMR requires query_embedding — call rerank_mmr() directly")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank_rrf([dummy], top_k=3)
    print("RRF results:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']}")
