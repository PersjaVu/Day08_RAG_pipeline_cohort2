# DATABASE — Conversation Memory (PostgreSQL)

> P3 — Bộ nhớ hội thoại cho chatbot LawAI.
> File này hướng dẫn **từ đầu tới có data**: tạo DB → bảng → index → seed → truy vấn lấy 10 chat gần nhất.

---

## 0. Yêu cầu

- PostgreSQL ≥ 13 (cần hàm `gen_random_uuid()` — có sẵn từ PG 13; nếu PG cũ hơn dùng extension `pgcrypto`).
- `pip install psycopg2-binary`
- Thêm vào `.env`:
  ```
  DATABASE_URL=postgresql://lawai_user:lawai_pass@localhost:5432/lawai
  ```

---

## 1. Tạo database + user (chạy bằng tài khoản postgres)

```sql
-- Kết nối psql với quyền superuser: psql -U postgres
CREATE DATABASE lawai;
CREATE USER lawai_user WITH PASSWORD 'lawai_pass';
GRANT ALL PRIVILEGES ON DATABASE lawai TO lawai_user;

-- Kết nối vào DB vừa tạo
\c lawai

-- (PG < 13) bật pgcrypto nếu chưa có gen_random_uuid
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Cấp quyền schema public cho user (PG 15+ cần dòng này)
GRANT ALL ON SCHEMA public TO lawai_user;
```

---

## 2. Tạo bảng (schema)

```sql
-- Bảng phiên hội thoại: mỗi cuộc trò chuyện = 1 session
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Bảng tin nhắn: mỗi lượt (user/assistant) thuộc 1 session
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL
                REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Quan hệ:** `chat_sessions (1) ──< chat_messages (N)` — xoá session thì xoá luôn tin nhắn (`ON DELETE CASCADE`).

---

## 3. Index (tăng tốc lấy tin nhắn gần nhất)

```sql
-- Truy vấn chính là "lấy N tin nhắn mới nhất của 1 session"
-- → index theo (session_id, created_at DESC) cho phép đọc cực nhanh
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON chat_messages (session_id, created_at DESC);
```

---

## 4. Seed data mẫu (để test)

```sql
-- Tạo 1 session và lấy id của nó
INSERT INTO chat_sessions (title)
VALUES ('Tư vấn hình phạt ma tuý')
RETURNING session_id;
-- giả sử trả về: '11111111-1111-1111-1111-111111111111'

-- Thêm vài lượt hội thoại vào session đó
INSERT INTO chat_messages (session_id, role, content) VALUES
('11111111-1111-1111-1111-111111111111', 'user',      'Tàng trữ trái phép chất ma tuý bị phạt thế nào?'),
('11111111-1111-1111-1111-111111111111', 'assistant', 'Theo Điều 249 Bộ luật Hình sự 2015...'),
('11111111-1111-1111-1111-111111111111', 'user',      'Thế còn người tái phạm thì sao?'),
('11111111-1111-1111-1111-111111111111', 'assistant', 'Tái phạm là tình tiết tăng nặng...');
```

---

## 5. Truy vấn dùng trong app

### 5.1. Lấy 10 tin nhắn GẦN NHẤT của 1 session (conversation memory)

```sql
-- Lấy 10 dòng mới nhất rồi sắp lại theo thời gian tăng dần (cũ → mới)
-- để đưa vào prompt LLM đúng trình tự hội thoại.
SELECT role, content
FROM (
    SELECT role, content, created_at
    FROM chat_messages
    WHERE session_id = '11111111-1111-1111-1111-111111111111'
    ORDER BY created_at DESC
    LIMIT 10
) AS recent
ORDER BY created_at ASC;
```

### 5.2. Lưu 1 tin nhắn mới

```sql
INSERT INTO chat_messages (session_id, role, content)
VALUES ('11111111-1111-1111-1111-111111111111', 'user', 'Câu hỏi tiếp theo...');
```

### 5.3. Liệt kê các session gần đây (cho sidebar)

```sql
SELECT session_id, title, created_at
FROM chat_sessions
ORDER BY created_at DESC
LIMIT 20;
```

---

## 6. Khởi tạo tự động bằng Python

Toàn bộ bước 2–3 đã được gói trong [memory/schema.sql](memory/schema.sql) và chạy tự động bởi
[memory/db.py](memory/db.py):

```python
from group_project.memory.db import init_db, create_session, save_message, get_recent_messages

init_db()                                  # tạo bảng + index nếu chưa có
sid = create_session("Phiên mới")          # -> session_id
save_message(sid, "user", "...")           # lưu lượt user
save_message(sid, "assistant", "...")      # lưu lượt assistant
history = get_recent_messages(sid, 10)     # 10 lượt gần nhất (memory)
```

> Nếu `DATABASE_URL` chưa cấu hình hoặc Postgres không chạy, `db.py` tự fallback sang
> bộ nhớ tạm (in-memory) để demo không vỡ — nhưng lịch sử sẽ mất khi tắt app.

---

## 7. Luồng tích hợp memory trong chatbot

```
User gửi câu hỏi (kèm session_id)
   │
   ├─ get_recent_messages(session_id, 10)   ← lấy 10 lượt gần nhất
   │
   ├─ rag_answer(query, history=10_luot)    ← P1: retrieve + generate có memory
   │
   ├─ save_message(session_id, 'user', query)
   └─ save_message(session_id, 'assistant', answer)
```
