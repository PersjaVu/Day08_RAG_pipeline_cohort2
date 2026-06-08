"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — dùng structural
understanding của document (cây mục lục / heading) thay vì embedding.

Cài đặt:
    pip install pageindex

API thật (pageindex>=0.x) dùng class PageIndexClient, làm việc trên file PDF
và theo cơ chế BẤT ĐỒNG BỘ:
    1. submit_document(pdf_path)        -> {'doc_id': ...}   (xử lý OCR + tree async)
    2. is_retrieval_ready(doc_id)       -> bool              (chờ tới khi sẵn sàng)
    3. submit_query(doc_id, query)      -> {'retrieval_id': ...}
    4. get_retrieval(retrieval_id)      -> {..., kết quả}    (poll tới khi xong)

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai → lấy API key → thêm PAGEINDEX_API_KEY vào .env
    2. Chạy `python src/task8_pageindex_vectorless.py` để upload PDF (chạy 1 lần)
    3. Dùng pageindex_search() làm fallback trong pipeline (Task 9)
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

# PageIndex làm việc trên PDF gốc → dùng các file trong data/landing/legal/
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
# Lưu lại doc_id sau khi upload để tái sử dụng (tránh upload lại tốn quota)
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

# Thời gian chờ tối đa (giây) cho xử lý async của PageIndex
UPLOAD_READY_TIMEOUT = 600   # chờ OCR + tree generation (file luật lớn + scan)
QUERY_TIMEOUT = 90           # chờ kết quả retrieval
POLL_INTERVAL = 5


def _is_configured() -> bool:
    """True nếu có API key thật (bỏ qua placeholder rỗng / chứa 'xxx')."""
    key = PAGEINDEX_API_KEY.strip()
    return bool(key) and "xxx" not in key.lower()


def _get_client():
    """Khởi tạo PageIndexClient từ SDK thật."""
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


# =============================================================================
# Upload documents (chạy 1 lần)
# =============================================================================

def upload_documents() -> list[str]:
    """
    Upload toàn bộ PDF trong data/landing/legal/ lên PageIndex.
    Chờ tới khi mỗi document sẵn sàng cho retrieval, lưu doc_id ra file.

    Returns:
        List doc_id (str).
    """
    if not _is_configured():
        raise RuntimeError("PAGEINDEX_API_KEY chưa được set thật trong .env")

    client = _get_client()
    pdfs = sorted(LANDING_LEGAL_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"  [pageindex] Không tìm thấy PDF trong {LANDING_LEGAL_DIR}")
        return []

    from pageindex import PageIndexAPIError

    doc_ids = []
    for pdf in pdfs:
        print(f"  [pageindex] Uploading: {pdf.name}")
        try:
            resp = client.submit_document(str(pdf))
        except PageIndexAPIError as exc:
            # Ví dụ {"detail":"LimitReached"} khi tài khoản hết quota → bỏ qua, chạy tiếp
            print(f"    [SKIP] {pdf.name}: {exc}")
            continue
        doc_id = resp.get("doc_id")
        if doc_id:
            doc_ids.append(doc_id)
            print(f"    -> doc_id={doc_id}")

    if not doc_ids:
        print("  [pageindex] Không upload được tài liệu nào (có thể hết quota tài khoản).")
        return []

    # Chờ tất cả document xử lý xong (OCR + tree generation)
    print("  [pageindex] Đang chờ xử lý (OCR + tree)...")
    deadline = time.monotonic() + UPLOAD_READY_TIMEOUT
    pending = set(doc_ids)
    while pending and time.monotonic() < deadline:
        for doc_id in list(pending):
            if client.is_retrieval_ready(doc_id):
                pending.discard(doc_id)
                print(f"    [ready] {doc_id}")
        if pending:
            time.sleep(POLL_INTERVAL)

    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
    print(f"  [pageindex] Đã lưu {len(doc_ids)} doc_id vào {DOC_IDS_FILE.name}")
    return doc_ids


def _load_doc_ids(client) -> list[str]:
    """Lấy doc_id đã upload: ưu tiên file local, fallback list_documents()."""
    if DOC_IDS_FILE.exists():
        ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        if ids:
            return ids
    # Fallback: hỏi trực tiếp API
    resp = client.list_documents(limit=100)
    docs = resp.get("documents", []) if isinstance(resp, dict) else []
    return [d.get("id") or d.get("doc_id") for d in docs if (d.get("id") or d.get("doc_id"))]


# =============================================================================
# Retrieval (async: submit_query -> poll get_retrieval)
# =============================================================================

def _wait_retrieval(client, retrieval_id: str) -> dict:
    """Poll get_retrieval tới khi hoàn tất hoặc hết thời gian."""
    deadline = time.monotonic() + QUERY_TIMEOUT
    last = {}
    while time.monotonic() < deadline:
        last = client.get_retrieval(retrieval_id)
        status = str(last.get("status", "")).lower()
        if status in ("completed", "done", "success", "finished", ""):
            # status rỗng: một số phiên bản trả kết quả ngay
            if status or last.get("retrieval") or last.get("results"):
                return last
        if status in ("failed", "error"):
            return last
        time.sleep(POLL_INTERVAL)
    return last


def _parse_results(payload: dict) -> list[dict]:
    """
    Trích danh sách chunk từ response của get_retrieval.

    Schema thật của PageIndex retrieval:
        {
          "retrieved_nodes": [
            {
              "id": "0005",
              "title": "Điều 5...",
              "metadata": [doc_id, filename, ...],
              "relevant_contents": [[{"relevant_content": "...", ...}]]  # list lồng
            }, ...
          ]
        }
    Lưu ý: nodes đã sắp theo độ liên quan giảm dần nhưng KHÔNG có score số
    → gán score giả lập theo rank để giữ format {content, score, metadata}.
    """
    nodes = payload.get("retrieved_nodes") or []
    n = len(nodes)
    parsed = []
    for rank, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue

        # Gom relevant_content từ cấu trúc lồng relevant_contents: list[list[dict]]
        texts = []
        for group in node.get("relevant_contents") or []:
            sub_items = group if isinstance(group, list) else [group]
            for item in sub_items:
                if isinstance(item, dict) and item.get("relevant_content"):
                    texts.append(item["relevant_content"])
        content = "\n\n".join(texts).strip() or node.get("title", "")
        if not content:
            continue

        # Score giả lập theo rank (node đầu = liên quan nhất → score cao nhất)
        score = round(1.0 - rank / max(n, 1), 4)

        meta = node.get("metadata")
        source_file = meta[1] if isinstance(meta, list) and len(meta) > 1 else ""
        parsed.append({
            "content": content,
            "score": score,
            "metadata": {
                "title": node.get("title", ""),
                "node_id": node.get("id", ""),
                "source": source_file,
                "doc_id": payload.get("doc_id", ""),
            },
            "source": "pageindex",
        })
    return parsed


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
        Trả về [] (graceful) nếu chưa cấu hình hoặc gặp lỗi.
    """
    if not _is_configured():
        print("  [pageindex] PAGEINDEX_API_KEY chưa cấu hình — trả về rỗng")
        return []

    try:
        client = _get_client()
        doc_ids = _load_doc_ids(client)
        if not doc_ids:
            print("  [pageindex] Chưa có document. Chạy upload_documents() trước.")
            return []

        all_results = []
        for doc_id in doc_ids:
            sub = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = sub.get("retrieval_id")
            if not retrieval_id:
                continue
            payload = _wait_retrieval(client, retrieval_id)
            all_results.extend(_parse_results(payload))

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    except Exception as exc:
        print(f"  [pageindex] Query failed: {exc}")
        return []


if __name__ == "__main__":
    if not _is_configured():
        print("PageIndex chưa cấu hình (PAGEINDEX_API_KEY rỗng/placeholder).")
        print("Đăng ký tại https://pageindex.ai/ → thêm key thật vào .env → pip install pageindex")
    else:
        print("Uploading documents (chạy 1 lần, có thể mất vài phút)...")
        try:
            ids = upload_documents()
        except Exception as exc:
            ids = []
            print(f"Upload lỗi: {exc}")

        if ids:
            print("\nTest query:")
            results = pageindex_search("hình phạt tàng trữ trái phép chất ma tuý", top_k=3)
            for r in results:
                print(f"[{r['score']:.3f}] {r['content'][:100]}...")
        else:
            print("Không có tài liệu nào được index (kiểm tra quota tài khoản PageIndex).")
