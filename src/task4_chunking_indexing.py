"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn kỹ thuật:
  Chunking: RecursiveCharacterTextSplitter
    - chunk_size=800: đủ ngữ cảnh cho văn bản pháp luật tiếng Việt (câu dài),
      không quá dài gây loãng embedding
    - chunk_overlap=100: ~12% overlap để không đứt đoạn câu quan trọng
    - separator=["\n\n","\n",". "," ",""]: ưu tiên xuống dòng trước khi cắt giữa câu

  Embedding: BAAI/bge-m3
    - 1024 dim, multilingual (tiếng Việt tốt), top leaderboard MTEB
    - Mô hình cục bộ, không cần API, miễn phí

  Vector Store: ChromaDB (persistent local)
    - Đơn giản, không cần Docker hay server riêng
    - Hỗ trợ cosine similarity, lưu metadata

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import json
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "vectorstore" / "chroma"
CORPUS_JSON = STANDARDIZED_DIR / "chunks.json"
COLLECTION_NAME = "drug_docs"

# chunk_size=800: văn bản pháp luật tiếng Việt có câu dài, cần đủ ngữ cảnh
CHUNK_SIZE = 800
# chunk_overlap=100: ~12% overlap tránh đứt đoạn ý tại ranh giới chunk
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# BAAI/bge-m3: model đa ngôn ngữ tốt nhất cho tiếng Việt, 1024 dim
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def get_embedding_model():
    """Load BAAI/bge-m3 một lần, cache ở module level."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[dict], model=None) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.

    Returns:
        chunks với key 'embedding' được thêm vào mỗi item.
    """
    if model is None:
        model = get_embedding_model()

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB (persistent local).
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Xóa collection cũ nếu tồn tại (để re-index sạch)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert theo batch (ChromaDB limit 5461/batch)
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        collection.add(
            ids=[f"chunk_{start + j}" for j in range(len(batch))],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
    print(f"  Indexed {len(chunks)} chunks into ChromaDB at {CHROMA_DIR}")


def save_corpus_for_bm25(chunks: list[dict]):
    """Lưu corpus (không embedding) sang JSON để Task 6 (BM25) dùng."""
    corpus = [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    CORPUS_JSON.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(corpus)} chunks to {CORPUS_JSON.name}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 55)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE}")
    print("=" * 55)

    docs = load_documents()
    print(f"\n[1/4] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[2/4] Created {len(chunks)} chunks")

    print(f"[3/4] Embedding with {EMBEDDING_MODEL} ...")
    model = get_embedding_model()
    chunks = embed_chunks(chunks, model)

    print("[4/4] Indexing ...")
    index_to_vectorstore(chunks)
    save_corpus_for_bm25(chunks)

    print(f"\nDone. {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    run_pipeline()
