# Bài Tập Nhóm — DrugLaw RAG Chatbot + Evaluation Pipeline

**Day 8 | RAG Pipeline v2 | Cohort 2**

---

## Sản Phẩm

Nhóm thực hiện **CẢ HAI** yêu cầu:

1. **RAG Chatbot** — Streamlit web app với conversation memory và citation
2. **RAG Evaluation** — Custom evaluation pipeline với 4 metrics + A/B comparison

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                               │
│                   Streamlit Chat (app.py)                           │
│              [Input] ←→ [Chat History] ←→ [Sources]                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │ query + history
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE (Task 9)                      │
│                                                                     │
│   ┌──────────────────┐    ┌──────────────────┐                      │
│   │  Semantic Search │    │  Lexical Search  │                      │
│   │  (ChromaDB +     │    │  (BM25Okapi)     │                      │
│   │  MiniLM-L6-v2)   │    │                  │                      │
│   └────────┬─────────┘    └────────┬─────────┘                      │
│            │                       │                                 │
│            └──────────┬────────────┘                                │
│                       ▼                                             │
│              ┌─────────────────┐                                    │
│              │  RRF Merge      │  Reciprocal Rank Fusion            │
│              └────────┬────────┘                                    │
│                       ▼                                             │
│              ┌─────────────────┐                                    │
│              │  Reranking      │  Cross-encoder keyword scoring      │
│              └────────┬────────┘                                    │
│                       │                                             │
│              score < threshold?                                     │
│                       │ YES                                         │
│                       ▼                                             │
│              ┌─────────────────┐                                    │
│              │  PageIndex      │  Vectorless fallback (Task 8)      │
│              │  Fallback       │                                    │
│              └─────────────────┘                                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │ top-k chunks
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATION (Task 10)                             │
│                                                                     │
│  1. reorder_for_llm() → tránh "lost in the middle"                  │
│  2. format_context() → thêm source labels cho citation              │
│  3. Build prompt với conversation history (memory)                  │
│  4. Call LLM (Claude / OpenAI) → Answer có citation                 │
└─────────────────────────────────────────────────────────────────────┘

DATA LAYER:
    data/landing/legal/    → 3 văn bản pháp luật DOCX (BLHS, Luật PCMT, NĐ 105)
    data/landing/news/     → 5 bài báo JSON về nghệ sĩ VN và ma tuý
    data/standardized/     → Markdown converted (Task 3)
    data/chroma_db/        → ChromaDB vector index (Task 4)
    data/chunks.json       → BM25 corpus (Task 4)
```

---

## Evaluation Architecture

```
Golden Dataset (18 Q&A pairs)
    ├── Category: legal_criminal (6 câu)
    ├── Category: legal_prevention (6 câu)
    └── Category: news_artists (6 câu)
              │
              ├── Config A: Hybrid + Rerank ──┐
              │                               ├── 4 Metrics ──→ A/B Report
              └── Config B: Dense-only ───────┘
                                               
Metrics (Custom Rule-Based):
    1. Faithfulness    = |answer_tokens ∩ context_tokens| / |answer_tokens|
    2. Answer Relevance = Jaccard(answer, question+expected)
    3. Context Recall   = |expected_tokens ∩ retrieved_tokens| / |expected_tokens|
    4. Context Precision = relevant_chunks / total_chunks
```

---

## Kết Quả Evaluation (tóm tắt)

| Config | Faithfulness | Relevance | Recall | Precision | **Average** |
|--------|-------------|-----------|--------|-----------|-------------|
| **A: Hybrid + Rerank** | **0.281** | 0.010 | 0.722 | 1.000 | **0.503** |
| B: Dense-only | 0.247 | 0.010 | 0.726 | 1.000 | 0.496 |
| **Δ** | **+0.035** | 0.000 | -0.003 | 0.000 | **+0.008** |

→ **Config A thắng** về Faithfulness và Overall Score.

Chi tiết: xem [evaluation/results.md](evaluation/results.md)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Store | ChromaDB (local persistent) |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Lexical Search | BM25Okapi (rank-bm25) |
| Merge | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder keyword scoring |
| Vectorless Fallback | PageIndex (BM25 fallback) |
| Generation | Claude Haiku / OpenAI GPT-4o-mini |
| UI | Streamlit |
| Evaluation | Custom Rule-Based Metrics |

---

## Cấu Trúc Files

```
group_project/
├── README.md                 ← File này
└── evaluation/
    ├── golden_dataset.json   ← 18 Q&A pairs (3 categories)
    ├── eval_pipeline.py      ← Script evaluation + A/B comparison
    └── results.md            ← Bảng điểm + phân tích

app.py                        ← Streamlit chatbot (root của project)
```

---

## Hướng Dẫn Chạy

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Tạo dữ liệu (nếu chưa có)
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown
python -m src.task4_chunking_indexing

# 3. Chạy chatbot
streamlit run app.py

# 4. Chạy evaluation
python group_project/evaluation/eval_pipeline.py
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Đinh Nguyễn Nhật Lâm | — | Xây dựng RAG Pipeline (Tasks 1-10) + Phát triển Chatbot UI | ✅ Hoàn thành |
| Tạ Văn Huấn | — | Thu thập & Chuẩn hoá dữ liệu (legal/news) + Thiết kế Golden Dataset | ✅ Hoàn thành |
| Trần Gia Huy | — | Tích hợp Reranking, Fallback PageIndex + So sánh A/B & Chạy Evaluation | ✅ Hoàn thành |
| Vũ Duy Bảo | — | Thiết kế Chunking & Indexing ChromaDB + Triển khai giao diện chatbot | ✅ Hoàn thành |
| Vũ Quang Bảo | — | Xây dựng bộ tìm kiếm Hybrid Search + Viết tài liệu Hướng dẫn sử dụng | ✅ Hoàn thành |
| Phạm Mạnh Thắng | — | Xây dựng các metrics đánh giá & phân tích worst performers | ✅ Hoàn thành |

---

## Lưu Ý

> Repo này sẽ được phát triển thêm ở Track 3 Giai đoạn 2 với **Knowledge Graph** để xử lý các câu hỏi phức tạp liên quan đến quan hệ giữa các điều luật và vụ việc thực tế.