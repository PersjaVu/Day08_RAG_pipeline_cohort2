"""
P2 — LawAI Chatbot (Flask).

Kiến trúc đúng mô hình:
    Flask UI → Retrieval (Task 9) → Generation (Task 10) → Display
               └──────── P1: rag_answer() ────────┘
    + P3: conversation memory (Postgres) — nhớ 10 lượt gần nhất theo session.

Chạy:
    pip install flask google-generativeai psycopg2-binary
    # .env: GEMINI_API_KEY=... (và tuỳ chọn DATABASE_URL=postgresql://...)
    python group_project/app.py
    # mở http://127.0.0.1:5000
"""

import io
import os
import sys
from pathlib import Path

# Cho phép import src.* và group_project.* khi chạy trực tiếp file này
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Model embedding đã được Task 4 tải về cache → ép offline, tránh HF Hub rate-limit
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from group_project.memory.db import (
    create_session,
    get_recent_messages,
    init_db,
    save_message,
)
from group_project.rag_pipeline import rag_answer

app = Flask(__name__)

# Khởi tạo DB (Postgres nếu có DATABASE_URL, không thì in-memory)
init_db()


def _format_sources(chunks: list[dict]) -> list[dict]:
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


@app.route("/api/session", methods=["POST"])
def new_session():
    """Tạo phiên chat mới, trả session_id cho frontend."""
    session_id = create_session(title="Cuộc trò chuyện mới")
    return jsonify({"session_id": session_id})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()
    session_id = data.get("session_id")
    if not query:
        return jsonify({"answer": "Vui lòng nhập câu hỏi.", "sources": []})

    # Tạo session nếu frontend chưa có
    if not session_id:
        session_id = create_session(title=query[:50])

    # P3 — lấy 10 lượt gần nhất làm memory
    history = get_recent_messages(session_id, limit=10)

    # P1 — UI → Task 9 retrieve → Task 10 generate (có memory)
    result = rag_answer(query, history=history)
    answer = result.get("answer", "")

    # P3 — lưu lượt user + assistant vào memory
    save_message(session_id, "user", query)
    save_message(session_id, "assistant", answer)

    return jsonify({
        "session_id": session_id,
        "answer": answer,
        "sources": _format_sources(result.get("sources", [])),
        "retrieval_source": result.get("retrieval_source", ""),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
