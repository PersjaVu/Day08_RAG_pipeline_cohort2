"""
Task 5 — Semantic Search Module.

Dùng ChromaDB + BAAI/bge-m3 (cùng model/store ở Task 4).
Query được embed, tìm top_k chunks gần nhất bằng cosine similarity.
"""

from pathlib import Path

_model = None   # lazy-loaded
_collection = None  # lazy-loaded

CHROMA_DIR = Path(__file__).parent.parent / "data" / "vectorstore" / "chroma"
COLLECTION_NAME = "drug_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng cosine similarity trên ChromaDB.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance ∈ [0,2]; similarity = 1 - distance/2 → [0,1]
        score = 1.0 - dist / 2.0
        output.append({"content": doc, "score": score, "metadata": meta})

    return sorted(output, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata'].get('source','?')}) {r['content'][:100]}...")
