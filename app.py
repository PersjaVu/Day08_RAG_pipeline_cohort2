"""
LawAI — RAG Chatbot tư vấn pháp luật về ma tuý (Bài nhóm — P2 Frontend).

Kiến trúc đúng theo mô hình gợi ý trong README:

    Flask UI  →  Retrieval (Task 9)  →  Generation (Task 10)  →  Display
                 └──────────── generate_with_citation() ───────────┘

App chỉ là lớp Display: nhận câu hỏi → gọi Task 10 (đã bao gồm Task 9 retrieve
bên trong) → trả answer + sources cho giao diện.

LLM: Gemini 2.5 Flash (ưu tiên) / OpenAI (fallback) — cấu hình trong Task 10.

Chạy:
    pip install flask google-generativeai
    # thêm GEMINI_API_KEY vào .env  (lấy tại https://aistudio.google.com/apikey)
    python app.py
    # mở http://127.0.0.1:5000
"""

import io
import sys

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

# Fix Windows console encoding để log tiếng Việt không crash
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Generation (Task 10) — bên trong đã gọi Retrieval (Task 9)
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation

app = Flask(__name__)


def _format_sources(chunks: list[dict]) -> list[dict]:
    """Chuyển chunks (Task 9/10) sang định dạng gọn cho giao diện."""
    return [
        {
            "source": c.get("metadata", {}).get("source", "N/A"),
            "type": c.get("metadata", {}).get("type", c.get("source", "")),
            "score": round(float(c.get("score", 0.0)), 4),
            "preview": c.get("content", "")[:200],
        }
        for c in chunks
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()
    if not query:
        return jsonify({"answer": "Vui lòng nhập câu hỏi.", "sources": []})

    # UI → Task 9 (retrieve) → Task 10 (generate) → Display
    try:
        result = generate_with_citation(query)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        retrieval_source = result.get("retrieval_source", "")
    except Exception as exc:
        # LLM lỗi (key sai / hết quota) → vẫn trả về tài liệu retrieve được
        sources = retrieve(query, top_k=5)
        answer = (f"⚠️ Chưa sinh được câu trả lời ({exc}). "
                  f"Dưới đây là các tài liệu liên quan nhất hệ thống tìm được.")
        retrieval_source = "retrieval-only"

    return jsonify({
        "answer": answer,
        "sources": _format_sources(sources),
        "retrieval_source": retrieval_source,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
