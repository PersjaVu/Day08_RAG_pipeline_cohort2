# 📊 RAG Evaluation Report — LawAI

> Đánh giá pipeline RAG tư vấn pháp luật ma tuý. Báo cáo gồm **2 phần**:
> phần **A — Retrieval** đo thật bằng local (không cần LLM); phần **B — LLM-judge**
> (DeepEval) đã có code + smoke-test chạy được, đang chờ hạn mức Gemini reset.

| | |
|---|---|
| **Ngày chạy** | 2026-06-08 |
| **Dataset** | 16 cặp Q&A ([golden_dataset.json](golden_dataset.json)) |
| **Embedding** | paraphrase-multilingual-MiniLM-L12-v2 (384-dim) |
| **Generation / Judge model** | Gemini 2.5 Flash |
| **Configs A/B** | `hybrid_rerank` (semantic+lexical+RRF) vs `dense_only` (chỉ semantic) |

---

## Phần A — Retrieval Quality (đo THẬT, local, 16/16 câu)

Retrieval (Task 9) chạy hoàn toàn local (MiniLM + BM25 + ChromaDB) nên đo được đầy đủ
không tốn quota. Script: [`retrieval_eval.py`](retrieval_eval.py) · số liệu thô:
[`retrieval_results.json`](retrieval_results.json).

| Metric | `hybrid_rerank` 🏆 | `dense_only` | Δ |
|--------|:---:|:---:|:---:|
| **Context Recall (proxy)** | **0.955** | 0.910 | **+0.045** |
| **Hit-rate@5** | 1.000 | 1.000 | 0.000 |
| Avg top-score | 0.0164¹ | 0.8509² | — |

¹ điểm RRF (thang ~1/(60+rank)); ² cosine similarity [0,1] → **không so trực tiếp được** giữa 2 config (khác thang đo).

- **Context Recall proxy** = % từ khoá trong `expected_context` xuất hiện ở các chunk lấy về.
- **Hit-rate@5** = tỉ lệ câu có ≥1 đoạn khớp tốt trong top-5.

### Phổ điểm recall (config tốt nhất `hybrid_rerank`)
```
Context Recall  █████████▌ 0.955
Hit-rate@5      ██████████ 1.000
```

### Worst performers (recall thấp nhất)

| Câu hỏi | `hybrid_rerank` | `dense_only` | Nhận xét |
|---------|:---:|:---:|---|
| Chất ma tuý được định nghĩa như thế nào...? | 0.75 | 0.75 | "định nghĩa" là khái niệm rải rác, khó khớp từ khoá |
| Các hành vi bị nghiêm cấm theo Luật PCMT 2021? | 0.875 | **0.458** | dense yếu hẳn — rerank+lexical kéo recall lên gần gấp đôi |
| Tội mua bán trái phép chất ma tuý...? | 0.80 | 0.80 | thuật ngữ "Điều 251" hiếm, cần lexical hỗ trợ |

> 13/16 câu đạt recall = 1.0 ở config `hybrid_rerank`.

---

## Phần B — LLM-judge Metrics (DeepEval) — ⏳ chờ quota

4 metric theo yêu cầu đề được implement đầy đủ trong [`eval_pipeline.py`](eval_pipeline.py):
**Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision** (judge = Gemini).

**Trạng thái:** code đã chạy được — smoke test 1 case cho `Faithfulness = 1.0` (thật). Tuy nhiên
**không thể chạy trọn 16 câu × 2 config hôm nay** vì Gemini **free tier giới hạn 20 request/ngày/model**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quota_value: 20), trong khi full eval cần ~250–300 lần gọi.

**Cách lấy số đầy đủ** (1 trong 2):
1. **Bật billing** Google AI Studio (pay-as-you-go, ~vài cent cho cả eval) → chạy:
   ```bash
   python group_project/evaluation/eval_pipeline.py
   ```
2. **Chờ 24h** quota free reset, rồi chạy lệnh trên (nên đặt `EVAL_LIMIT=6` để vừa 20 req/ngày).

Khi chạy, `eval_pipeline.py` tự ghi đè bảng dưới + lưu `results_raw.json`:

| Config | Faithfulness | Answer Relevancy | Contextual Recall | Contextual Precision | Avg |
|--------|:---:|:---:|:---:|:---:|:---:|
| `hybrid_rerank` | _chờ_ | _chờ_ | _chờ_ | _chờ_ | _chờ_ |
| `dense_only` | _chờ_ | _chờ_ | _chờ_ | _chờ_ | _chờ_ |

---

## Phần C — Phân tích & Đề xuất

**Kết luận A/B (dựa trên retrieval thật):**
- `hybrid_rerank` **tốt hơn** `dense_only` về recall (0.955 vs 0.910). Khác biệt rõ nhất ở các câu
  giàu thuật ngữ pháp lý ("nghiêm cấm", số Điều) — nơi **lexical BM25 + RRF** bù cho điểm yếu của
  dense embedding. → **Nên dùng hybrid + rerank làm mặc định.**
- Cả hai đạt hit-rate@5 = 1.0 → retriever luôn lấy được ít nhất 1 đoạn liên quan; vấn đề còn lại là
  *xếp hạng* và *độ đầy đủ*, đúng chỗ rerank phát huy.

**Đề xuất cải tiến:**

| Triệu chứng (quan sát thật) | Nguyên nhân | Hành động |
|---|---|---|
| Câu "định nghĩa chất ma tuý" recall 0.75 ở cả 2 config | Khái niệm trải dài nhiều câu, 1 chunk không đủ | Tăng `chunk_size`/overlap, hoặc `top_k` lên 8 |
| dense_only tụt mạnh ở câu "hành vi bị nghiêm cấm" (0.458) | Embedding bỏ lỡ từ khoá hiếm | Giữ lexical trong pipeline (đừng dùng dense-only) |
| Câu hỏi theo số Điều ("Điều 251/255") | Dense kém với mã số | Ưu tiên BM25 cho truy vấn chứa số điều |
| Faithfulness/Answer Relevancy chưa đo | Hết quota free tier | Bật billing hoặc chạy lại sau 24h |

> Số liệu thô retrieval: [`retrieval_results.json`](retrieval_results.json).
> Khi có quota, số LLM-judge sẽ tự điền vào Phần B + [`results_raw.json`](results_raw.json).
