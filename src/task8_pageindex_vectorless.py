"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — dùng structural
understanding của document (headings, tables, lists) thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key → thêm PAGEINDEX_API_KEY vào .env
    3. Chạy upload_documents() một lần
    4. Dùng pageindex_search() trong pipeline
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Index ID trả về sau khi upload (lưu lại để tái sử dụng)
_INDEX_ID: str | None = None


def upload_documents() -> str:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Returns:
        index_id (str) — dùng cho query sau.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được set trong .env")

    from pageindex import PageIndex

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)

    docs = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        docs.append({
            "content": content,
            "metadata": {
                "filename": md_file.name,
                "type": md_file.parent.name,
            },
        })

    print(f"  Uploading {len(docs)} documents to PageIndex...")
    index = pi.create_index(documents=docs)
    index_id = index.id

    global _INDEX_ID
    _INDEX_ID = index_id
    print(f"  [OK] Index created: {index_id}")
    return index_id


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if not PAGEINDEX_API_KEY:
        print("  [pageindex] PAGEINDEX_API_KEY not set — returning empty results")
        return []

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        index_id = _INDEX_ID
        if not index_id:
            # Lấy index đầu tiên nếu chưa upload trong session này
            indices = pi.list_indices()
            if not indices:
                print("  [pageindex] No index found. Run upload_documents() first.")
                return []
            index_id = indices[0].id

        results = pi.query(index_id=index_id, query=query, top_k=top_k)

        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": getattr(r, "metadata", {}),
                "source": "pageindex",
            }
            for r in results
        ]

    except Exception as exc:
        print(f"  [pageindex] Query failed: {exc}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Set PAGEINDEX_API_KEY trong .env, sau đó chạy lại.")
        print("Dang ky tai: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
