import argparse
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pilot results with RAGAS using local LLM and embeddings.",
    )
    parser.add_argument(
        "--rag-report",
        default="evaluation-pilot/rag-pilot-report.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation-pilot/ragas-report.json",
    )
    parser.add_argument(
        "--evaluator-model",
        default=os.getenv("RAGAS_LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:3b")),
    )
    parser.add_argument(
        "--evaluator-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
    )
    args = parser.parse_args()

    rag_report_path = Path(args.rag_report)
    if not rag_report_path.is_file():
        raise SystemExit(
            f"Relatório RAG não encontrado em '{rag_report_path}'. "
            "Execute primeiro: python -m paa_rag evaluate --dataset evaluation-pilot/questions-pt-rag.jsonl "
            "--output evaluation-pilot/rag-pilot-report.json"
        )

    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as error:
        raise SystemExit(
            "Dependências do RAGAS ausentes. Instale com: "
            "pip install -r evaluation/requirements.txt\n"
            f"Detalhes: {error}"
        )

    report = json.loads(rag_report_path.read_text(encoding="utf-8"))
    rows = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    for item in report.get("results", []):
        rows["user_input"].append(str(item["question"]))
        rows["response"].append(str(item["answer"]))
        rows["retrieved_contexts"].append([str(c) for c in item.get("contexts", [])])
        rows["reference"].append(str(item.get("ground_truth") or ""))

    if not rows["user_input"]:
        raise SystemExit("O relatório RAG não contém resultados para avaliar.")

    evaluator_llm = ChatOllama(
        model=args.evaluator_model,
        base_url=args.evaluator_url,
        temperature=0.0,
    )
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]
    ragas_result = evaluate(
        Dataset.from_dict(rows),
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    score_rows = ragas_result.scores
    metric_names = list(score_rows[0]) if score_rows else []
    summary: dict[str, float | None] = {}
    per_question: list[dict[str, float | None]] = []

    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in score_rows
            if isinstance(row.get(metric_name), (int, float))
            and not math.isnan(float(row[metric_name]))
        ]
        summary[metric_name] = (sum(values) / len(values)) if values else None

    for row in score_rows:
        per_question.append({
            name: (
                float(value)
                if isinstance(value, (int, float)) and not math.isnan(float(value))
                else None
            )
            for name, value in row.items()
        })

    output = {
        "evaluator_model": args.evaluator_model,
        "embedding_model": args.embedding_model,
        "questions": len(rows["user_input"]),
        "ragas": per_question,
        "summary": summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"\n[OK] Relatório RAGAS salvo em {output_path.resolve()}")


if __name__ == "__main__":
    main()