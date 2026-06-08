"""
P3 — Conversation Memory (PostgreSQL).

Lưu mỗi phiên chat (session_id) cùng toàn bộ tin nhắn trong phiên.
Khi chat, lấy 10 tin nhắn gần nhất của session để LLM "nhớ" ngữ cảnh.

Cấu hình .env:
    DATABASE_URL=postgresql://user:password@localhost:5432/lawai

Cài đặt:
    pip install psycopg2-binary

Nếu Postgres không khả dụng → tự fallback sang bộ nhớ tạm (in-memory) để demo
không bị vỡ; log cảnh báo rõ ràng.
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
SCHEMA_FILE = Path(__file__).parent / "schema.sql"
RECENT_LIMIT = 10  # số tin nhắn gần nhất dùng làm memory

# ---- Fallback in-memory (khi không có Postgres) ----
_mem_store: dict[str, list[dict]] = {}
_use_postgres = False


def _connect():
    """Mở kết nối Postgres. Raise nếu lỗi → caller quyết định fallback."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def init_db() -> bool:
    """
    Khởi tạo schema. Trả True nếu dùng được Postgres, False nếu fallback in-memory.
    """
    global _use_postgres
    if not DATABASE_URL:
        print("  [memory] DATABASE_URL chưa set — dùng bộ nhớ tạm (in-memory).")
        _use_postgres = False
        return False
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
            conn.commit()
        _use_postgres = True
        print("  [memory] PostgreSQL sẵn sàng.")
        return True
    except Exception as exc:
        print(f"  [memory] Không kết nối được Postgres ({exc}) — dùng in-memory.")
        _use_postgres = False
        return False


def create_session(title: str | None = None) -> str:
    """Tạo phiên chat mới, trả session_id."""
    if _use_postgres:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (title) VALUES (%s) RETURNING session_id;",
                (title,),
            )
            session_id = str(cur.fetchone()[0])
            conn.commit()
            return session_id
    # in-memory
    session_id = str(uuid.uuid4())
    _mem_store[session_id] = []
    return session_id


def save_message(session_id: str, role: str, content: str) -> None:
    """Lưu 1 tin nhắn (role = 'user' | 'assistant') vào phiên."""
    if _use_postgres:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s);",
                (session_id, role, content),
            )
            conn.commit()
        return
    # in-memory
    _mem_store.setdefault(session_id, []).append({"role": role, "content": content})


def get_recent_messages(session_id: str, limit: int = RECENT_LIMIT) -> list[dict]:
    """
    Lấy `limit` tin nhắn GẦN NHẤT của session, trả theo thứ tự thời gian tăng dần
    (cũ → mới) để đưa vào prompt làm conversation memory.
    """
    if _use_postgres:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) AS recent
                ORDER BY created_at ASC;
                """,
                (session_id, limit),
            )
            return [{"role": r, "content": c} for r, c in cur.fetchall()]
    # in-memory
    msgs = _mem_store.get(session_id, [])
    return msgs[-limit:]


if __name__ == "__main__":
    import sys
    if isinstance(sys.stdout, __import__("io").TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    init_db()
    sid = create_session("Phiên test")
    print("session:", sid)
    save_message(sid, "user", "Hình phạt tàng trữ ma tuý?")
    save_message(sid, "assistant", "Theo Điều 249 BLHS...")
    for m in get_recent_messages(sid):
        print(f"  {m['role']}: {m['content'][:50]}")
