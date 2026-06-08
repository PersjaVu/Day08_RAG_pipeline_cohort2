"""
P4 — RAG Evaluation Pipeline (DeepEval).

Đánh giá pipeline RAG bằng DeepEval với 4 metrics, dùng Gemini 2.5 Flash làm judge.
So sánh A/B 2 config retrieval và xuất báo cáo ra results.md (P5).

Cài đặt:
    pip install deepeval
    # .env: GEMINI_API_KEY=...  (judge model)

Chạy:
    python group_project/evaluation/eval_pipeline.py

Metrics:
    - Faithfulness        : câu trả lời có bám đúng context không?
    - Answer Relevancy    : câu trả lời có đúng câu hỏi không?
    - Contextual Recall   : retriever có lấy đủ evidence (so với expected_answer)?
    - Contextual Precision: context lấy về có bao nhiêu % thực sự hữu ích?

A/B configs:
    - hybrid_rerank : semantic + lexical + rerank (mặc định)
    - dense_only    : chỉ semantic, không rerank
"""

import io
import json
import os
import sys
from pathlib import Path

# Cho phép import src.* khi chạy trực tiếp
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Model embedding đã cache → offline, tránh HF Hub rate-limit
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.task9_retrieval_pipeline import semantic_search  # dense-only config
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    SYSTEM_PROMPT,
    _call_gemini,
    _call_openai,
    format_context,
    reorder_for_llm,
)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Groq: free tier hào phóng, OpenAI-compatible → dùng làm generation + judge
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
USE_GROQ = bool(os.getenv("GROQ_API_KEY"))

CONFIGS = {
    "hybrid_rerank": "Semantic + Lexical + RRF rerank (mặc định)",
    "dense_only": "Chỉ semantic search, không rerank",
}


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


# =============================================================================
# Sinh câu trả lời theo từng config (để A/B)
# =============================================================================

def _retry(fn, attempts: int = 8):
    """
    Gọi lại fn khi gặp lỗi tạm thời:
      - 429 / quota / ResourceExhausted (free tier rate limit) → chờ ~30s
      - 503 / UNAVAILABLE (quá tải) → chờ ~15s
    """
    import time
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            if "429" in msg or "quota" in msg or "exhausted" in msg or "rate" in msg:
                wait = 30
            elif "503" in msg or "unavailable" in msg:
                wait = 15
            else:
                wait = 5 * (i + 1)
            print(f"    [retry {i+1}/{attempts}] {str(exc)[:70]} — chờ {wait}s")
            time.sleep(wait)
    raise last


def _call_groq(user_message: str) -> str:
    """Generation qua Groq (OpenAI-compatible)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        top_p=0.9,
    )
    return resp.choices[0].message.content


def _llm(user_message: str) -> str:
    if USE_GROQ:
        return _retry(lambda: _call_groq(user_message))
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return _retry(lambda: _call_gemini(user_message))
    return _retry(lambda: _call_openai(user_message))


def generate_for_config(question: str, config: str, top_k: int = 5) -> tuple[str, list[str]]:
    """Trả về (answer, retrieval_context) theo config retrieval."""
    if config == "dense_only":
        chunks = semantic_search(question, top_k=top_k)
    else:  # hybrid_rerank
        chunks = retrieve(question, top_k=top_k)

    context = format_context(reorder_for_llm(chunks))
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
    answer = _llm(user_message)
    return answer, [c["content"] for c in chunks]


# =============================================================================
# DeepEval — judge bằng Gemini
# =============================================================================

def _get_judge():
    """Tạo judge model cho DeepEval (Gemini 2.5 Flash). None → dùng mặc định (OpenAI)."""
    # Ưu tiên Groq (free tier rộng, OpenAI-compatible) làm judge
    if USE_GROQ:
        try:
            from deepeval.models import LocalModel
            return LocalModel(
                model=GROQ_MODEL,
                api_key=os.getenv("GROQ_API_KEY"),
                base_url=GROQ_BASE_URL,
            )
        except Exception as exc:
            print(f"  [eval] Không tạo được Groq judge ({exc}) — thử Gemini.")

    judge_model = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
    try:
        from deepeval.models import GeminiModel
        return GeminiModel(
            model=judge_model,
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        )
    except Exception as exc:
        print(f"  [eval] Không tạo được GeminiModel ({exc}) — dùng judge mặc định.")
        return None


def evaluate_config(config: str, dataset: list[dict]) -> dict:
    """
    Chạy 4 metric DeepEval cho 1 config trên toàn bộ golden dataset.
    Trả về {metric: avg_score} + danh sách per-case để phân tích worst performers.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge = _get_judge()
    mk = lambda M: M(threshold=0.7, model=judge) if judge else M(threshold=0.7)
    metric_defs = {
        "faithfulness": lambda: mk(FaithfulnessMetric),
        "answer_relevancy": lambda: mk(AnswerRelevancyMetric),
        "contextual_recall": lambda: mk(ContextualRecallMetric),
        "contextual_precision": lambda: mk(ContextualPrecisionMetric),
    }

    per_case = []
    print(f"\n=== Config: {config} — {CONFIGS[config]} ===")
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        answer, contexts = generate_for_config(q, config)
        tc = LLMTestCase(
            input=q,
            actual_output=answer,
            expected_output=item["expected_answer"],
            retrieval_context=contexts,
        )
        import time
        scores = {}
        for name, factory in metric_defs.items():
            try:
                def _measure():
                    m = factory()
                    m.measure(tc)
                    return float(m.score or 0.0)
                scores[name] = _retry(_measure)
            except Exception as exc:
                print(f"    [warn] {name} lỗi ở câu {i}: {str(exc)[:80]}")
                scores[name] = 0.0
            time.sleep(5)  # giãn nhịp tránh vượt rate limit free tier
        per_case.append({"question": q, "scores": scores})
        avg = sum(scores.values()) / len(scores)
        print(f"  [{i:02d}/{len(dataset)}] avg={avg:.2f}  {q[:50]}")
        # Lưu tăng dần để không mất data nếu bị dừng giữa chừng
        partial = RESULTS_PATH.parent / f"partial_{config}.json"
        partial.write_text(json.dumps(per_case, ensure_ascii=False, indent=2), encoding="utf-8")

    # Trung bình từng metric
    agg = {}
    for name in metric_defs:
        vals = [c["scores"][name] for c in per_case]
        agg[name] = sum(vals) / len(vals) if vals else 0.0
    return {"aggregate": agg, "per_case": per_case}


# =============================================================================
# A/B + Export
# =============================================================================

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "contextual_recall": "Contextual Recall",
    "contextual_precision": "Contextual Precision",
}


def _bar(score: float, width: int = 10) -> str:
    """Thanh tiến độ bằng ký tự khối cho dễ nhìn."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def export_results(results_by_config: dict, timestamp: str = ""):
    """Ghi báo cáo so sánh A/B + worst/best performers + đề xuất ra results.md (P5)."""
    metrics = list(METRIC_LABELS.keys())
    n_cases = len(next(iter(results_by_config.values()))["per_case"])

    # Lưu kết quả thô để minh bạch / tái dùng
    raw_path = RESULTS_PATH.parent / "results_raw.json"
    raw_path.write_text(json.dumps(results_by_config, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = {c: sum(r["aggregate"].values()) / len(r["aggregate"])
               for c, r in results_by_config.items()}
    best_cfg = max(overall, key=overall.get)

    L = []
    L.append("# 📊 RAG Evaluation Report — LawAI")
    L.append("")
    L.append("> Đánh giá tự động pipeline RAG tư vấn pháp luật ma tuý.")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    _model_name = GROQ_MODEL if USE_GROQ else os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
    L.append("| **Framework** | DeepEval |")
    L.append(f"| **Judge model** | {_model_name} |")
    L.append(f"| **Generation model** | {_model_name} |")
    L.append("| **Embedding** | paraphrase-multilingual-MiniLM-L12-v2 (384-dim) |")
    L.append(f"| **Golden dataset** | {n_cases} cặp Q&A |")
    L.append(f"| **Configs A/B** | {', '.join(f'`{c}`' for c in results_by_config)} |")
    if timestamp:
        L.append(f"| **Thời điểm chạy** | {timestamp} |")
    L.append("")

    # ----- 1. Metrics là gì -----
    L.append("## 1. Metrics đánh giá")
    L.append("")
    L.append("| Metric | Ý nghĩa |")
    L.append("|--------|---------|")
    L.append("| **Faithfulness** | Câu trả lời có bám đúng context không (chống bịa)? |")
    L.append("| **Answer Relevancy** | Câu trả lời có đúng trọng tâm câu hỏi không? |")
    L.append("| **Contextual Recall** | Retriever có lấy đủ evidence (so với đáp án chuẩn)? |")
    L.append("| **Contextual Precision** | % chunk lấy về thực sự hữu ích (xếp hạng đúng)? |")
    L.append("")
    L.append("Cấu hình A/B:")
    for c in results_by_config:
        L.append(f"- **`{c}`** — {CONFIGS[c]}")
    L.append("")

    # ----- 2. Bảng A/B -----
    L.append("## 2. Kết quả A/B")
    L.append("")
    header = "| Config | " + " | ".join(METRIC_LABELS[m] for m in metrics) + " | **Avg** |"
    L.append(header)
    L.append("|" + "---|" * (len(metrics) + 2))
    for cfg, res in results_by_config.items():
        agg = res["aggregate"]
        row = [f"{agg[m]:.3f}" for m in metrics]
        star = " 🏆" if cfg == best_cfg else ""
        L.append(f"| `{cfg}`{star} | " + " | ".join(row) + f" | **{overall[cfg]:.3f}** |")
    L.append("")

    # Winner mỗi metric + delta
    if len(results_by_config) == 2:
        cfgs = list(results_by_config.keys())
        a, b = cfgs[0], cfgs[1]
        L.append(f"**Chênh lệch theo từng metric ( `{a}` − `{b}` ):**")
        L.append("")
        L.append("| Metric | " + f"`{a}`" + " | " + f"`{b}`" + " | Δ | Thắng |")
        L.append("|---|---|---|---|---|")
        for m in metrics:
            va, vb = results_by_config[a]["aggregate"][m], results_by_config[b]["aggregate"][m]
            d = va - vb
            win = a if d > 0 else (b if d < 0 else "hoà")
            L.append(f"| {METRIC_LABELS[m]} | {va:.3f} | {vb:.3f} | {d:+.3f} | `{win}` |")
        L.append("")

    # ----- 3. Biểu đồ điểm config tốt nhất -----
    L.append(f"## 3. Phổ điểm — config tốt nhất `{best_cfg}`")
    L.append("")
    agg_best = results_by_config[best_cfg]["aggregate"]
    L.append("```")
    for m in metrics:
        L.append(f"{METRIC_LABELS[m]:<22} {_bar(agg_best[m])} {agg_best[m]:.3f}")
    L.append("```")
    L.append("")

    per_case = results_by_config[best_cfg]["per_case"]
    ranked = sorted(per_case, key=lambda c: sum(c["scores"].values()) / len(c["scores"]))

    # ----- 4. Worst performers -----
    L.append("## 4. Worst performers (5 câu thấp nhất)")
    L.append("")
    L.append("| # | Câu hỏi | Faith | AnsRel | Recall | Prec | Chẩn đoán |")
    L.append("|---|---------|-------|--------|--------|------|-----------|")
    for i, c in enumerate(ranked[:5], 1):
        s = c["scores"]
        diag = _diagnose(s)
        L.append(f"| {i} | {c['question'][:48]} | {s['faithfulness']:.2f} | "
                 f"{s['answer_relevancy']:.2f} | {s['contextual_recall']:.2f} | "
                 f"{s['contextual_precision']:.2f} | {diag} |")
    L.append("")

    # ----- 5. Best performers -----
    L.append("## 5. Best performers (3 câu cao nhất)")
    L.append("")
    L.append("| # | Câu hỏi | Avg |")
    L.append("|---|---------|-----|")
    for i, c in enumerate(reversed(ranked[-3:]), 1):
        avg = sum(c["scores"].values()) / len(c["scores"])
        L.append(f"| {i} | {c['question'][:60]} | {avg:.3f} |")
    L.append("")

    # ----- 6. Phân tích & đề xuất -----
    L.append("## 6. Phân tích & đề xuất cải tiến")
    L.append("")
    L.append(f"- **Config tốt nhất:** `{best_cfg}` (Avg = {overall[best_cfg]:.3f}).")
    weakest = min(agg_best, key=agg_best.get)
    L.append(f"- **Metric yếu nhất:** {METRIC_LABELS[weakest]} ({agg_best[weakest]:.3f}).")
    L.append("")
    L.append("| Triệu chứng | Nguyên nhân | Hành động đề xuất |")
    L.append("|---|---|---|")
    L.append("| Contextual Recall thấp | Retriever bỏ sót evidence | Tăng `top_k`, cải thiện chunking, bật PageIndex fallback |")
    L.append("| Faithfulness thấp, AnsRel cao | LLM bịa ngoài context | Siết SYSTEM_PROMPT, hạ temperature |")
    L.append("| Contextual Precision thấp | Nhiều chunk nhiễu | Tăng cường rerank (cross-encoder), giảm `top_k` |")
    L.append("| Answer Relevancy thấp | Lạc đề / trả lời chung chung | Cải thiện prompt, thêm few-shot |")
    L.append("")
    L.append("> Số liệu thô đầy đủ: [`results_raw.json`](results_raw.json)")
    L.append("")

    RESULTS_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[OK] Đã ghi báo cáo vào {RESULTS_PATH}")
    print(f"[OK] Số liệu thô: {raw_path}")


def _diagnose(s: dict) -> str:
    """Sinh chẩn đoán ngắn cho 1 câu dựa trên điểm các metric."""
    if s["contextual_recall"] < 0.5:
        return "Thiếu evidence (recall thấp)"
    if s["faithfulness"] < 0.5:
        return "Có dấu hiệu bịa (faithfulness thấp)"
    if s["contextual_precision"] < 0.5:
        return "Context nhiễu (precision thấp)"
    if s["answer_relevancy"] < 0.5:
        return "Trả lời lạc trọng tâm"
    return "Ổn định"


def main():
    dataset = load_golden_dataset()
    # EVAL_LIMIT: giới hạn số câu để chạy thật trong hạn mức free tier (mặc định: tất cả)
    limit = int(os.getenv("EVAL_LIMIT", "0"))
    if limit > 0:
        dataset = dataset[:limit]
    print(f"Loaded {len(dataset)} test cases (limit={limit or 'all'})")

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("⚠ Cần GEMINI_API_KEY (hoặc OPENAI_API_KEY) trong .env để chạy LLM + judge.")
        return

    results_by_config = {}
    for cfg in CONFIGS:
        results_by_config[cfg] = evaluate_config(cfg, dataset)

    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    export_results(results_by_config, timestamp=ts)


if __name__ == "__main__":
    main()
