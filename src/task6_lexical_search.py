"""
Task 6 — Lexical Search Module (BM25).

Sử dụng BM25Okapi (rank-bm25).

Cơ chế BM25:
  score(q,d) = Σ IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
  - k1=1.5: điều chỉnh mức độ bão hoà tần suất từ
  - b=0.75: chuẩn hoá theo độ dài document
  Từ hiếm (IDF cao) trong tài liệu ngắn sẽ được điểm cao.

Corpus được load từ data/standardized/chunks.json (Task 4 đã tạo).

Cài đặt:
    pip install rank-bm25
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CORPUS_JSON = Path(__file__).parent.parent / "data" / "standardized" / "chunks.json"

_bm25: BM25Okapi | None = None
_corpus: list[dict] = []


def _load_corpus() -> list[dict]:
    if not CORPUS_JSON.exists():
        raise FileNotFoundError(
            f"{CORPUS_JSON} chưa tồn tại. Hãy chạy task4_chunking_indexing.py trước."
        )
    return json.loads(CORPUS_JSON.read_text(encoding="utf-8"))


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.
    Tokenize đơn giản bằng whitespace — tiếng Việt không cần underthesea
    cho mức độ chính xác cơ bản vì từ đã tách sẵn.
    """
    tokenized = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized)


def _ensure_index():
    global _bm25, _corpus
    if _bm25 is None:
        _corpus = _load_corpus()
        _bm25 = build_bm25_index(_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khoá sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    _ensure_index()

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": _corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata'].get('source','?')}) {r['content'][:100]}...")
