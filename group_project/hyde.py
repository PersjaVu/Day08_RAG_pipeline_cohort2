"""
P6 (Bonus) — HyDE: Hypothetical Document Embeddings.

Ý tưởng (Gao et al. 2022): câu hỏi ngắn thường khác xa về từ ngữ với đoạn văn bản
chứa đáp án → embedding câu hỏi không khớp tốt. HyDE khắc phục bằng cách:

    1. Cho LLM sinh một "câu trả lời giả định" (hypothetical document) cho câu hỏi.
    2. Embed CHÍNH câu trả lời giả định đó (giàu thuật ngữ pháp lý) để tìm kiếm,
       thay vì embed câu hỏi gốc → tăng recall vì gần với văn bản luật hơn.

Tái dùng: Task 5 semantic_search (embed + ChromaDB) + LLM của Task 10.

So sánh nhanh:
    semantic_search(query)        → embed câu hỏi
    hyde_search(query)            → embed câu-trả-lời-giả-định của câu hỏi
"""

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.task5_semantic_search import semantic_search
from src.task10_generation import _call_gemini, _call_openai

HYDE_PROMPT = (
    "Bạn là chuyên gia pháp luật Việt Nam về phòng chống ma tuý. "
    "Hãy viết MỘT đoạn văn ngắn (3-5 câu) trả lời giả định cho câu hỏi dưới đây, "
    "dùng đúng văn phong và thuật ngữ của văn bản pháp luật (điều, khoản, hình phạt...). "
    "Không cần chính xác tuyệt đối — mục tiêu là tạo đoạn văn giống tài liệu luật để tìm kiếm.\n\n"
    "Câu hỏi: {query}\n\nĐoạn trả lời giả định:"
)


def generate_hypothetical(query: str) -> str:
    """Sinh 'tài liệu giả định' cho câu hỏi bằng LLM."""
    prompt = HYDE_PROMPT.format(query=query)
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return _call_gemini(prompt)
    return _call_openai(prompt)


def hyde_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieval theo HyDE: sinh tài liệu giả định → embed nó → semantic search.

    Returns: list[{content, score, metadata}] (giống semantic_search).
    """
    try:
        hypo = generate_hypothetical(query)
    except Exception as exc:
        print(f"  [hyde] Không sinh được tài liệu giả định ({exc}) — fallback semantic thường.")
        return semantic_search(query, top_k=top_k)

    # Ghép câu hỏi + tài liệu giả định để giữ cả ý gốc lẫn thuật ngữ luật
    enriched = f"{query}\n{hypo}"
    return semantic_search(enriched, top_k=top_k)


if __name__ == "__main__":
    q = "Người tái nghiện ma tuý bị xử lý thế nào?"
    print(f"Query: {q}\n")
    print("--- Tài liệu giả định (HyDE) ---")
    print(generate_hypothetical(q)[:400])
    print("\n--- Kết quả hyde_search ---")
    for i, r in enumerate(hyde_search(q, top_k=3), 1):
        print(f"{i}. [{r['score']:.3f}] {r['content'][:90]}")
