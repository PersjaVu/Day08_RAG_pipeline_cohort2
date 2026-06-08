"""
P1 — Integration Pipeline (xương sống bài nhóm).

Gộp bài cá nhân của các thành viên thành 1 interface thống nhất:

    rag_answer(query, history) :  UI → Retrieval (Task 9) → Generation (Task 10) → kết quả

- Retrieval: src.task9_retrieval_pipeline.retrieve  (semantic + lexical + rerank + fallback)
- Generation: tái dùng helper của Task 10 (reorder chống lost-in-the-middle,
  format context, SYSTEM_PROMPT) + LLM Gemini 2.5 Flash (ưu tiên) / OpenAI (fallback).
- Hỗ trợ conversation memory: truyền `history` (list các lượt gần nhất) để nhớ ngữ cảnh.

Đây là module mà P2 (chatbot) và P4/P5 (evaluation) cùng import dùng.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    SYSTEM_PROMPT,
    TOP_K,
    _call_gemini,
    _call_openai,
    format_context,
    reorder_for_llm,
)


def _has_llm() -> bool:
    return bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def _call_llm(user_message: str) -> str:
    """Gọi LLM: Gemini 2.5 Flash ưu tiên, fallback OpenAI."""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return _call_gemini(user_message)
    return _call_openai(user_message)


def _format_history(history: list[dict] | None) -> str:
    """Ghép lịch sử hội thoại thành text (conversation memory)."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "Người dùng" if turn.get("role") == "user" else "LawAI"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def rag_answer(
    query: str,
    history: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    Pipeline RAG hoàn chỉnh cho bài nhóm.

    Args:
        query: câu hỏi hiện tại
        history: danh sách lượt hội thoại trước [{role, content}] (memory)
        top_k: số chunk lấy về

    Returns:
        {
            'answer': str,
            'sources': list[dict],         # các chunk dùng (content, score, metadata)
            'retrieval_source': str,       # 'hybrid' | 'pageindex' | ...
        }
    """
    # Step 1 — Retrieval (Task 9)
    chunks = retrieve(query, top_k=top_k)

    # Step 2 — Reorder chống lost-in-the-middle + format context (Task 10)
    context = format_context(reorder_for_llm(chunks))

    # Step 3 — Build prompt (kèm conversation memory nếu có)
    history_text = _format_history(history)
    user_message = (
        (f"Lịch sử hội thoại gần đây:\n{history_text}\n\n" if history_text else "")
        + f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"
    )

    # Step 4 — Generation (LLM)
    if not _has_llm():
        answer = (
            "⚠️ Chưa cấu hình GEMINI_API_KEY (hoặc OPENAI_API_KEY) trong .env. "
            "Dưới đây là các tài liệu liên quan nhất hệ thống truy xuất được."
        )
    else:
        try:
            answer = _call_llm(user_message)
        except Exception as exc:
            answer = f"⚠️ Lỗi khi gọi LLM: {exc}. Xem các nguồn bên dưới."

    retrieval_src = chunks[0].get("source", "hybrid") if chunks else "none"
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_src,
    }


if __name__ == "__main__":
    import sys

    if isinstance(sys.stdout, __import__("io").TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    res = rag_answer("Hình phạt cho tội tàng trữ trái phép chất ma tuý?")
    print("ANSWER:\n", res["answer"][:500])
    print(f"\n[{len(res['sources'])} nguồn | via {res['retrieval_source']}]")
