-- ============================================================================
-- P3 — Conversation Memory schema (PostgreSQL)
-- Lưu phiên chat (session) và toàn bộ tin nhắn trong phiên đó.
-- ============================================================================

-- Bảng phiên hội thoại
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Bảng tin nhắn — mỗi dòng là 1 lượt (user hoặc assistant) thuộc 1 session
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index để lấy nhanh N tin nhắn gần nhất của 1 session
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON chat_messages (session_id, created_at DESC);
