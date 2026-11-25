import argparse
import json
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create the frozen Portuguese RAG evaluation dataset.",
    )
    parser.add_argument(
        "--questions",
        default="evaluation-pilot/questions-pt.jsonl",
    )
    parser.add_argument(
        "--chunks",
        default="corpus-pilot/chunks.jsonl",
    )
    parser.add_argument(
        "--answers",
        default="evaluation-pilot/expected-answers-pt.jsonl",
    )
    parser.add_argument(
        "--output",
        default="evaluation-pilot/questions-pt-rag.jsonl",
    )
    args = parser.parse_args()

    answers = {
        item["question"]: item["expected_answer"]
        for item in (
            json.loads(line)
            for line in Path(args.answers).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    chunks = {}
    with Path(args.chunks).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["language"] == "pt":
                chunks.setdefault(record["title"], []).append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with Path(args.questions).open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["question"] not in answers:
                raise SystemExit(
                    f"Missing expected answer for: {item['question']}"
                )
            references = []
            relevant_ids = []
            for title in item["relevant_titles"]:
                if title not in chunks:
                    raise SystemExit(f"Title absent from pilot: {title}")
                record = chunks[title][0]
                relevant_ids.append(record["chunk_id"])
                references.append({
                    "chunk_id": record["chunk_id"],
                    "title": record["title"],
                    "url": record["url"],
                    "text": record["text"][:900],
                })

            record = {
                "question": item["question"],
                "relevant_chunk_ids": relevant_ids,
                "contexts": [reference["text"] for reference in references],
                "ground_truth": answers[item["question"]],
                "references": references,
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "count": count,
        "output": str(output_path),
        "sha256": digest,
        "source_questions": args.questions,
        "source_chunks": args.chunks,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
