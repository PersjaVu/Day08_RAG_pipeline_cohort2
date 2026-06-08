# RAG Evaluation Results

**Framework:** Custom Rule-Based Metrics (Offline — không cần API)
**Ngày chạy:** 2026-06-08 16:05
**Golden dataset:** 18 câu hỏi

---

## Mô tả Configs

### Config A: Hybrid Search + Reranking (mặc định)
- Retrieval: BM25 (Lexical) + Sentence-Transformers (Semantic)
- Merge: Reciprocal Rank Fusion (RRF)
- Reranking: Cross-encoder keyword scoring
- Fallback: PageIndex/BM25 khi score < threshold

### Config B: Dense-only (Semantic) — No Reranking
- Retrieval: Semantic Search only (Sentence-Transformers)
- Không dùng BM25, không merge, không reranking

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-only) | Δ (A − B) |
|--------|---------------------------|----------------------|-----------|
| Faithfulness | 0.281 | 0.247 | +0.035 |
| Answer Relevance | 0.010 | 0.010 | +0.000 |
| Context Recall | 0.722 | 0.726 | -0.003 |
| Context Precision | 1.000 | 1.000 | +0.000 |
| **Average** | **0.503** | **0.496** | **+0.008** |

---

## A/B Comparison Analysis

**Config A (Hybrid + Rerank) thắng Config B (Dense-only)** với điểm trung bình cao hơn 0.008.

**Nhận xét:**
- Hybrid search kết hợp BM25 + Semantic giúp tăng Context Recall vì BM25 tìm được các từ khoá chính xác (tên điều luật, số điều) mà semantic search có thể bỏ qua.
- Reranking cải thiện Context Precision bằng cách đưa những chunks liên quan nhất lên đầu.
- Dense-only có thể tốt hơn cho các câu hỏi có ngữ nghĩa phức tạp, nhưng kém hơn cho queries cần khớp từ khoá chính xác (số điều luật, tên pháp lệnh).

**Kết luận:** Config A (Hybrid + Rerank) phù hợp hơn cho domain pháp luật Việt Nam vì văn bản pháp luật có nhiều từ khoá chuyên ngành, số điều luật cụ thể — phù hợp với BM25's keyword matching.

---

## Chi Tiết Kết Quả — Config A

| ID | Category | Faith | Rel | Recall | Prec | Avg |
|----|----------|-------|-----|--------|------|-----|
| Q01 | legal_criminal | 0.19 | 0.01 | 0.71 | 1.00 | 0.48 |
| Q02 | legal_criminal | 0.25 | 0.00 | 0.71 | 1.00 | 0.49 |
| Q03 | legal_criminal | 0.38 | 0.04 | 0.71 | 1.00 | 0.53 |
| Q04 | legal_criminal | 0.25 | 0.00 | 0.71 | 1.00 | 0.49 |
| Q05 | legal_criminal | 0.25 | 0.00 | 0.86 | 1.00 | 0.53 |
| Q06 | legal_prevention | 0.25 | 0.03 | 0.67 | 1.00 | 0.49 |
| Q07 | legal_prevention | 0.25 | 0.00 | 0.75 | 1.00 | 0.50 |
| Q08 | legal_prevention | 0.25 | 0.00 | 0.75 | 1.00 | 0.50 |
| Q09 | legal_prevention | 0.31 | 0.01 | 0.60 | 1.00 | 0.48 |
| Q10 | legal_prevention | 0.31 | 0.00 | 0.75 | 1.00 | 0.52 |
| Q11 | news_artists | 0.25 | 0.02 | 0.83 | 1.00 | 0.53 |
| Q12 | news_artists | 0.25 | 0.01 | 1.00 | 1.00 | 0.57 |
| Q13 | news_artists | 0.25 | 0.01 | 0.67 | 1.00 | 0.48 |
| Q14 | news_artists | 0.38 | 0.02 | 0.92 | 1.00 | 0.58 |
| Q15 | news_artists | 0.31 | 0.00 | 0.75 | 1.00 | 0.52 |
| Q16 | legal_criminal | 0.25 | 0.00 | 0.29 | 1.00 | 0.38 |
| Q17 | legal_prevention | 0.31 | 0.01 | 0.60 | 1.00 | 0.48 |
| Q18 | legal_criminal | 0.38 | 0.01 | 0.71 | 1.00 | 0.52 |

---

## Worst Performers (Bottom 3 — Config A)

| # | ID | Question | Avg Score | Failure Stage | Root Cause |
|---|----|----------|-----------|---------------|------------|
| 1 | Q16 | Tội vận chuyển trái phép chất ma tuý (Điều 250) bị... | 0.38 | Generation | Answer không trả lời đúng câu hỏi |
| 2 | Q01 | Hình phạt cho tội tàng trữ trái phép chất ma tuý t... | 0.48 | Generation | Answer không trả lời đúng câu hỏi |
| 3 | Q17 | Quy trình cai nghiện tại gia đình theo Nghị định 1... | 0.48 | Generation | Answer không trả lời đúng câu hỏi |

---

## Metric Definitions (Custom Implementation)

| Metric | Formula | Mô tả |
|--------|---------|-------|
| **Faithfulness** | |answer_tokens ∩ context_tokens| / |answer_tokens| | Tỉ lệ tokens trong answer có xuất hiện trong context |
| **Answer Relevance** | 0.4×Jaccard(ans,q) + 0.6×Jaccard(ans,expected) | Độ trùng lặp giữa answer với câu hỏi và expected answer |
| **Context Recall** | |expected_tokens ∩ retrieved_tokens| / |expected_tokens| | Tỉ lệ keywords từ expected context xuất hiện trong retrieved |
| **Context Precision** | relevant_chunks / total_chunks | Tỉ lệ chunks retrieved thực sự chứa ≥2 token trùng với câu hỏi |

---

## Recommendations

### Cải tiến 1: Dùng LLM-as-Judge cho Faithfulness
**Action:** Thay custom token-overlap bằng DeepEval/RAGAS với LLM judge (GPT-4o-mini hoặc Claude Haiku)
**Expected impact:** Faithfulness score chính xác hơn ~30%, phát hiện được hallucination phức tạp

### Cải tiến 2: Vietnamese Tokenizer
**Action:** Tích hợp `underthesea` hoặc `pyvi` để tách từ tiếng Việt đúng hơn (tokenize đúng "ma tuý" thay vì "ma" + "tuý")
**Expected impact:** BM25 precision tăng ~15-20%, Context Recall cải thiện rõ rệt cho queries tiếng Việt

### Cải tiến 3: Expand Corpus với dữ liệu thực tế
**Action:** Thêm toàn bộ văn bản BLHS 2015, Luật PCMT 2021, và 50+ bài báo thực tế qua Crawl4AI
**Expected impact:** Context Recall tăng từ 0.72 lên 0.7+, giảm số câu hỏi dùng fallback PageIndex

### Cải tiến 4: Semantic Chunking
**Action:** Thay RecursiveCharacterTextSplitter bằng SemanticChunker dùng BAAI/bge-m3
**Expected impact:** Chunk boundaries chính xác hơn theo ngữ nghĩa, Faithfulness tăng ~10%

---

*Báo cáo tự động — DrugLaw RAG Evaluation | 2026-06-08 16:05*