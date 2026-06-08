"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Logic:
    Query
      ├→ Semantic Search (Task 5)  ──┐
      │                               ├→ RRF Merge → Rerank → Results
      ├→ Lexical Search (Task 6)  ──┘
      │
      └→ Nếu best score < threshold → Fallback: PageIndex (Task 8)
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
# Dùng "rrf" mặc định vì không cần API key.
# Đổi sang "cross_encoder" nếu có JINA_API_KEY.
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        1. Semantic search + Lexical search (top_k*2 mỗi loại)
        2. Merge bằng RRF
        3. Rerank (mặc định RRF, có thể đổi sang cross-encoder)
        4. Nếu top score < threshold → fallback PageIndex
        5. Return top_k results

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
    """
    # Step 1: Chạy song song semantic + lexical
    candidates_per_ranker = top_k * 3
    dense_results = semantic_search(query, top_k=candidates_per_ranker)
    sparse_results = lexical_search(query, top_k=candidates_per_ranker)

    for item in dense_results:
        item.setdefault("source", "semantic")
    for item in sparse_results:
        item.setdefault("source", "lexical")

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=candidates_per_ranker)
    for item in merged:
        item["source"] = "hybrid"

    if not merged:
        print("  [pipeline] No hybrid results — falling back to PageIndex")
        return pageindex_search(query, top_k=top_k)

    # Step 3: Rerank
    if use_reranking:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: Fallback nếu best score dưới ngưỡng
    top_score = final_results[0]["score"] if final_results else 0.0
    if top_score < score_threshold:
        print(
            f"  [pipeline] Top score {top_score:.3f} < threshold {score_threshold} "
            f"— falling back to PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
