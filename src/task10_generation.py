"""
Task 10 — Generation Có Citation.

Lựa chọn tham số:
  top_k=5  : Đủ evidence mà không quá dài gây lost-in-the-middle
  top_p=0.9: Nucleus sampling — diverse nhưng không lan man
  temperature=0.3: RAG cần factual, giảm sáng tạo (hallucination)

LLM: gpt-4o-mini (OpenAI) — nhanh, rẻ, đủ chất lượng cho task retrieval-augmented.
Fallback: gpt-3.5-turbo nếu gpt-4o-mini không khả dụng.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

# top_k=5: nghiên cứu Lost-in-the-Middle (Liu et al. 2023) cho thấy
# LLM nhớ tốt nhất 3-5 chunks, thêm hơn làm giảm chất lượng
TOP_K = 5

# top_p=0.9: giữ sự linh hoạt ngôn ngữ nhưng loại bỏ token xác suất thấp
TOP_P = 0.9

# temperature=0.3: factual QA cần độ chính xác cao, không cần creative
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# Document Reordering (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, kém ở GIỮA.
    Strategy: chunk quan trọng nhất (rank 1) ở đầu,
              quan trọng thứ 2 ở cuối, còn lại ở giữa.

    Input (by score desc): [A, B, C, D, E]
    Output:                [A, C, E, D, B]
    """
    if len(chunks) <= 2:
        return chunks

    # Tách ra: chỉ số lẻ (0,2,4,...) → đầu; chỉ số chẵn (1,3,5,...) → cuối
    top_half = chunks[::2]       # index 0, 2, 4 → most important, go first
    bot_half = chunks[1::2][::-1]  # index 1, 3, 5 reversed → second-most at end
    return top_half + bot_half


# =============================================================================
# Context Formatting
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string với source label để LLM cite."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


# =============================================================================
# Generation
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str   # 'hybrid' hoặc 'pageindex'
        }
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "answer": "OPENAI_API_KEY chưa được set trong .env. Hãy thêm key và thử lại.",
            "sources": [],
            "retrieval_source": "none",
        }

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder (lost-in-the-middle prevention)
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content
    retrieval_src = chunks[0].get("source", "hybrid") if chunks else "none"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_src,
    }


if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]
    for q in queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
