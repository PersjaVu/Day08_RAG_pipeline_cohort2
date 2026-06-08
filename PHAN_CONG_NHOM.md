# Phân Công Bài Nhóm — RAG Pipeline (6 người)

> Tài liệu phân chia công việc cho bài nhóm Ngày 8, dựa trên yêu cầu trong [README.md](README.md).
> Bài cá nhân (Task 1–10) làm riêng từng người; bài nhóm là **ghép lại + xây sản phẩm hoàn chỉnh**.

---

## 0. Lưu ý chiến lược (đọc trước khi chia)

README viết *"xây dựng **1 trong 2** sản phẩm"*, **nhưng bảng chấm điểm bài nhóm (30đ) chấm CẢ HAI**:

| Thành phần | Điểm |
|-----------|------|
| Chatbot demo hoạt động | 8 |
| Tích hợp pipeline các thành viên | 4 |
| Kiến trúc rõ ràng + README | 3 |
| Chất lượng câu trả lời (citation, đúng nội dung) | 3 |
| **Evaluation pipeline** (DeepEval/RAGAS/TruLens) | **12** |
| → Golden dataset ≥15 Q&A | 3 |
| → Eval ≥4 metrics | 4 |
| → So sánh A/B ≥2 configs | 3 |
| → Báo cáo + worst performers | 2 |

**→ Với 6 người, làm CẢ Chatbot + Evaluation để ăn trọn 30đ, nhắm thêm Bonus 20đ.**

### Bonus (20đ) — phân bổ kèm vai trò bên dưới
| Tiêu chí | Điểm |
|----------|------|
| Giải thích cơ chế lexical search khác BM25 (trong demo) | 5 |
| Implement HyDE (Hypothetical Document Embeddings) | 5 |
| Deploy chatbot online (HF Spaces/Render) | 4 |
| Conversation memory (multi-turn) | 3 |
| UI/UX chất lượng (hiển thị source, score, highlight) | 3 |

---

## 1. Bảng phân công 6 người

| # | Vai trò | Việc cụ thể | Deliverable | Điểm nhắm |
|---|---------|-------------|-------------|-----------|
| **P1** | **Integration Lead** (xương sống) | Chọn pipeline tốt nhất từ bài cá nhân của 6 người → gộp thành 1 module `rag_pipeline` chung (retrieve + generate), chuẩn hoá interface, dựng vector store chung | Pipeline `src/` thống nhất | Tích hợp **4đ** |
| **P2** | **Chatbot / Frontend** | App chat (Streamlit/Chainlit), hiển thị câu trả lời + **source documents + score** + citation | `app.py` | Chatbot **8đ** + UI/UX bonus **3đ** |
| **P3** | **Conversation + Deploy** | Conversation memory (hỏi nối tiếp multi-turn) + deploy online (HF Spaces/Render) | App deploy được | Bonus **3đ + 4đ** |
| **P4** | **Eval: Dataset + Metrics** | Soạn `golden_dataset.json` (≥15 cặp Q&A: question / expected_answer / expected_context) + `eval_pipeline.py` chạy ≥4 metrics | 2 file eval | Eval **3đ + 4đ** |
| **P5** | **Eval: A/B + Báo cáo** | Chạy eval trên ≥2 config (vd: có rerank vs không, hybrid vs dense-only) + viết `results.md` (bảng điểm + phân tích worst performers + đề xuất cải tiến) | `results.md` | Eval **3đ + 2đ** |
| **P6** | **Kiến trúc + README + Bonus** | Vẽ sơ đồ kiến trúc, hoàn thiện `group_project/README.md` (điền bảng phân công), làm **HyDE** + chuẩn bị phần *giải thích lexical search khác BM25* cho demo | README + HyDE | Kiến trúc **3đ** + Bonus **5đ + 5đ** |

**Tổng nhắm:** Bài nhóm 30đ + Bonus ~20đ.

---

## 2. Thứ tự & phụ thuộc (để không bị kẹt)

```
Ngày 1 — Cả nhóm họp:
  • Chốt: dùng pipeline của AI cho từng Task (retrieval, rerank, generation...)
  • Giao P1 bắt đầu tích hợp
  • P4 soạn golden dataset NGAY (không cần chờ pipeline)
  • P6 dựng khung README + diagram NGAY

P1 hoàn thành "rag_pipeline" chung ──┬──→ P2 build chatbot lên trên
                                     ├──→ P3 thêm memory + deploy
                                     └──→ P5 chạy A/B eval (cần pipeline + dataset P4)
```

- **Điểm nghẽn:** P1 (tích hợp) là nền móng — P2/P3/P5 đều chờ. P1 nên ra **interface tối thiểu sớm** để mọi người bám vào, hoàn thiện sau.
- **Chạy song song ngay (không chờ P1):** golden dataset (P4), README + diagram (P6).

---

## 3. Chi tiết deliverable từng mảng

### P1 — Integration
- 1 module pipeline chung, hàm `retrieve(query, top_k)` và `generate_with_citation(query)`.
- Đảm bảo vector store (ChromaDB) build từ dữ liệu chung của nhóm.
- Viết hướng dẫn ngắn để P2–P5 import dùng.

### P2 — Chatbot
- Giao diện chat, ô nhập câu hỏi → hiển thị câu trả lời.
- Khung "Nguồn tham khảo": liệt kê source document + score của mỗi chunk dùng.
- Citation hiển thị dạng `[Nguồn, Năm]`.

### P3 — Conversation + Deploy
- Lưu lịch sử hội thoại → hỗ trợ câu hỏi nối tiếp ("còn điều khoản nào khác không?").
- Deploy lên Hugging Face Spaces hoặc Render → có link chạy online.

### P4 — Golden Dataset + Eval
- `group_project/evaluation/golden_dataset.json`: ≥15 cặp `{question, expected_answer, expected_context}` về pháp luật ma tuý + tin tức nghệ sĩ.
- `group_project/evaluation/eval_pipeline.py`: chạy 4 metrics — Faithfulness, Answer Relevance, Context Recall, Context Precision.

### P5 — A/B + Report
- Chạy eval với ≥2 config khác nhau (rerank on/off, hybrid vs dense-only).
- `group_project/evaluation/results.md`: bảng điểm so sánh + phân tích câu tệ nhất (worst performers) + đề xuất cải tiến.

### P6 — Kiến trúc + README + Bonus
- Sơ đồ kiến trúc hệ thống (data → index → retrieve → rerank → generate → UI).
- Điền bảng phân công trong `group_project/README.md`.
- HyDE: sinh câu trả lời giả định trước khi embed query → tăng recall.
- Slide/giải thích: lexical search khác gì BM25 (TF-IDF, hybrid built-in...).

---

## 4. Checklist nộp bài (Deliverables)

- [ ] Pipeline tích hợp chạy được (P1)
- [ ] `app.py` — chatbot demo (P2)
- [ ] Conversation memory + link deploy (P3)
- [ ] `group_project/evaluation/golden_dataset.json` — ≥15 Q&A (P4)
- [ ] `group_project/evaluation/eval_pipeline.py` (P4)
- [ ] `group_project/evaluation/results.md` — A/B + phân tích (P5)
- [ ] Sơ đồ kiến trúc + `group_project/README.md` điền đủ phân công (P6)
- [ ] Code push lên repository chung của nhóm
- [ ] Demo chạy được trong buổi trình bày

---

## 5. Bảng điền thông tin thành viên (P6 cập nhật)

| Người | Họ tên | MSSV | Vai trò | Trạng thái |
|-------|--------|------|---------|------------|
| P1 | | | Integration Lead | ⬜ |
| P2 | | | Chatbot / Frontend | ⬜ |
| P3 | | | Conversation + Deploy | ⬜ |
| P4 | | | Eval: Dataset + Metrics | ⬜ |
| P5 | | | Eval: A/B + Báo cáo | ⬜ |
| P6 | | | Kiến trúc + README + Bonus | ⬜ |
