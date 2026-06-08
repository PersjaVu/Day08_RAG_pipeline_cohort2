# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
                 ┌──────────────────────────────────────────────┐
   Người dùng ──▶│   P2 — LawAI Chatbot (Flask, giao diện Gemini)│
                 └───────────────┬──────────────────────────────┘
                                 │ /api/chat (session_id, message)
              ┌──────────────────┼───────────────────────────────┐
              ▼                                                    ▼
   ┌────────────────────┐                        ┌────────────────────────────┐
   │ P3 — Memory (PG)   │  10 lượt gần nhất       │ P1 — rag_answer()          │
   │ chat_sessions      │ ─────────────────────▶  │                            │
   │ chat_messages      │ ◀───── lưu lượt mới      │  Task 9 retrieve           │
   └────────────────────┘                         │   (semantic+lexical+rerank │
                                                   │    + PageIndex fallback)   │
                                                   │  Task 10 generate          │
                                                   │   (reorder + Gemini 2.5)   │
                                                   └─────────────┬──────────────┘
                                                                 ▼
                                                   answer + sources (citation)
                                                                 │
   P4/P5 — Evaluation (DeepEval, 4 metrics, A/B) ◀───────────────┘
   P6 — HyDE (tăng recall) · Architecture · README
```

**Luồng chính (đúng mô hình đề):** `Flask UI → Retrieval (Task 9) → Generation (Task 10) → Display`.

---

## Triển khai thực tế (P1–P6)

| Mảng | File | Nội dung |
|------|------|----------|
| **P1** Integration | [`rag_pipeline.py`](rag_pipeline.py) | `rag_answer(query, history)` gộp Task 9 retrieve + Task 10 generate, nhận memory |
| **P2** Chatbot | [`app.py`](app.py), [`templates/`](templates), [`static/`](static) | Flask + UI kiểu Gemini (LawAI): sidebar thu gọn, model Gemini 2.5 Flash, hiển thị nguồn + score |
| **P3** Memory | [`memory/db.py`](memory/db.py), [`memory/schema.sql`](memory/schema.sql), [`DATABASE.md`](DATABASE.md) | PostgreSQL: session + tin nhắn, lấy 10 lượt gần nhất; fallback in-memory |
| **P4** Eval | [`evaluation/golden_dataset.json`](evaluation/golden_dataset.json), [`evaluation/eval_pipeline.py`](evaluation/eval_pipeline.py) | 16 Q&A + DeepEval 4 metrics (judge Gemini), A/B 2 config |
| **P5** Report | [`evaluation/results.md`](evaluation/results.md) | Bảng A/B + worst performers + đề xuất (tự sinh từ eval) |
| **P6** Bonus | [`hyde.py`](hyde.py) | HyDE — sinh tài liệu giả định rồi embed để tăng recall |

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| (P1) | | Integration pipeline `rag_pipeline.py` | ✅ |
| (P2) | | Chatbot LawAI (Flask UI) | ✅ |
| (P3) | | Conversation memory (PostgreSQL) | ✅ |
| (P4) | | Golden dataset + eval pipeline | ✅ |
| (P5) | | A/B comparison + báo cáo | ✅ |
| (P6) | | HyDE + kiến trúc + README | ✅ |

> Điền tên + MSSV thật của từng thành viên vào bảng trên.

---

## Hướng Dẫn Chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Cấu hình .env (tối thiểu cần GEMINI_API_KEY)
#    GEMINI_API_KEY=...
#    DATABASE_URL=postgresql://user:pass@localhost:5432/lawai   # tuỳ chọn (P3)

# 3. Đảm bảo đã index dữ liệu (Task 4) — tạo ChromaDB + chunks.json
python src/task4_chunking_indexing.py

# 4a. Chạy chatbot LawAI (P2)
python group_project/app.py        # mở http://127.0.0.1:5000

# 4b. Chạy evaluation (P4/P5) — sinh results.md
python group_project/evaluation/eval_pipeline.py

# 4c. Thử HyDE (P6)
python group_project/hyde.py
```

**Ghi chú:**
- LLM dùng **Gemini 2.5 Flash** (`google.generativeai` sẽ cảnh báo deprecation nhưng vẫn chạy; có thể chuyển sang `google-genai` sau).
- Nếu chưa cấu hình `DATABASE_URL`, P3 tự dùng bộ nhớ tạm (in-memory) — demo vẫn chạy, mất lịch sử khi tắt app.
- Embedding offline (`HF_HUB_OFFLINE=1`) dùng model đã cache từ Task 4, tránh HF Hub rate-limit.

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
