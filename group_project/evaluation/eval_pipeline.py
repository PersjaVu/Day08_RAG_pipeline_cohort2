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

CONFIGS = {
    "hybrid_rerank": "Semantic + Lexical + RRF rerank (mặc định)",
    "dense_only": "Chỉ semantic search, không rerank",
}


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


# =============================================================================
# Sinh câu trả lời theo từng config (để A/B)
# =============================================================================

def _llm(user_message: str) -> str:
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return _call_gemini(user_message)
    return _call_openai(user_message)


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
    try:
        from deepeval.models import GeminiModel
        return GeminiModel(
            model_name="gemini-2.5-flash",
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
        scores = {}
        for name, factory in metric_defs.items():
            m = factory()
            try:
                m.measure(tc)
                scores[name] = float(m.score or 0.0)
            except Exception as exc:
                print(f"    [warn] {name} lỗi ở câu {i}: {exc}")
                scores[name] = 0.0
        per_case.append({"question": q, "scores": scores})
        avg = sum(scores.values()) / len(scores)
        print(f"  [{i:02d}/{len(dataset)}] avg={avg:.2f}  {q[:50]}")

    # Trung bình từng metric
    agg = {}
    for name in metric_defs:
        vals = [c["scores"][name] for c in per_case]
        agg[name] = sum(vals) / len(vals) if vals else 0.0
    return {"aggregate": agg, "per_case": per_case}


# =============================================================================
# A/B + Export
# =============================================================================

def export_results(results_by_config: dict):
    """Ghi báo cáo so sánh A/B + worst performers + đề xuất ra results.md (P5)."""
    lines = ["# RAG Evaluation Results\n"]
    lines.append("> Đánh giá bằng DeepEval (judge: Gemini 2.5 Flash) trên golden dataset "
                 f"{len(next(iter(results_by_config.values()))['per_case'])} câu.\n")

    # Bảng so sánh A/B
    lines.append("\n## 1. So sánh A/B các config\n")
    metrics = ["faithfulness", "answer_relevancy", "contextual_recall", "contextual_precision"]
    header = "| Config | " + " | ".join(metrics) + " | **Trung bình** |"
    sep = "|" + "---|" * (len(metrics) + 2)
    lines.append(header)
    lines.append(sep)
    for cfg, res in results_by_config.items():
        agg = res["aggregate"]
        row_vals = [f"{agg[m]:.2f}" for m in metrics]
        overall = sum(agg.values()) / len(agg)
        lines.append(f"| `{cfg}` | " + " | ".join(row_vals) + f" | **{overall:.2f}** |")

    # Worst performers (theo config tốt nhất)
    best_cfg = max(results_by_config, key=lambda c: sum(results_by_config[c]["aggregate"].values()))
    lines.append(f"\n## 2. Worst performers (config `{best_cfg}`)\n")
    per_case = results_by_config[best_cfg]["per_case"]
    ranked = sorted(per_case, key=lambda c: sum(c["scores"].values()) / len(c["scores"]))
    lines.append("| Câu hỏi | faith | ans_rel | ctx_recall | ctx_prec |")
    lines.append("|---|---|---|---|---|")
    for c in ranked[:5]:
        s = c["scores"]
        lines.append(
            f"| {c['question'][:55]} | {s['faithfulness']:.2f} | {s['answer_relevancy']:.2f} "
            f"| {s['contextual_recall']:.2f} | {s['contextual_precision']:.2f} |"
        )

    # Đề xuất
    lines.append("\n## 3. Phân tích & đề xuất cải tiến\n")
    lines.append(f"- Config tốt nhất theo điểm trung bình: **`{best_cfg}`**.")
    lines.append("- Câu điểm thấp thường do retriever lấy thiếu evidence (contextual_recall thấp) "
                 "→ tăng `top_k`, cải thiện chunking, hoặc bật PageIndex fallback.")
    lines.append("- Nếu faithfulness thấp nhưng answer_relevancy cao → mô hình bịa thêm ngoài context "
                 "→ siết SYSTEM_PROMPT, hạ temperature.")
    lines.append("- Nếu contextual_precision thấp → nhiều chunk nhiễu → tăng cường rerank / giảm top_k.")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] Đã ghi báo cáo vào {RESULTS_PATH}")


def main():
    dataset = load_golden_dataset()
    print(f"Loaded {len(dataset)} test cases")

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("⚠ Cần GEMINI_API_KEY (hoặc OPENAI_API_KEY) trong .env để chạy LLM + judge.")
        return

    results_by_config = {}
    for cfg in CONFIGS:
        results_by_config[cfg] = evaluate_config(cfg, dataset)

    export_results(results_by_config)


if __name__ == "__main__":
    main()
