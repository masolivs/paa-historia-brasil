import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(
        description="Generate normalized E5 embeddings for a corpus pilot.",
    )
    parser.add_argument("--chunks", default="corpus-pilot/chunks.jsonl")
    parser.add_argument("--output", default="embeddings-pilot/e5-small")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model)
    dimension = model.get_sentence_embedding_dimension()
    count = 0
    with chunks_path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                count += 1
                if args.limit and count >= args.limit:
                    break

    vectors_path = output_dir / "vectors.f32.npy"
    vectors = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype="float32",
        shape=(count, dimension),
    )
    ids_path = output_dir / "chunk_ids.jsonl"
    metadata_path = output_dir / "metadata.jsonl"

    texts = []
    records = []
    write_index = 0
    with chunks_path.open("r", encoding="utf-8") as source, \
         ids_path.open("w", encoding="utf-8") as ids_out, \
         metadata_path.open("w", encoding="utf-8") as metadata_out:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            texts.append("passage: " + record["text"])
            records.append(record)
            if len(texts) == args.batch_size:
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                vectors[write_index:write_index + len(embeddings)] = embeddings
                write_index += len(embeddings)
                for item in records:
                    ids_out.write(json.dumps(item["chunk_id"]) + "\n")
                    metadata_out.write(json.dumps({
                        "chunk_id": item["chunk_id"],
                        "document_id": item["document_id"],
                        "source": item["source"],
                        "language": item["language"],
                        "title": item["title"],
                        "url": item["url"],
                        "chunk_index": item["chunk_index"],
                        "text_hash": item["text_hash"],
                    }, ensure_ascii=False) + "\n")
                texts = []
                records = []
                if args.limit and write_index >= args.limit:
                    break

    if texts:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors[write_index:write_index + len(embeddings)] = embeddings
        with ids_path.open("a", encoding="utf-8") as ids_out, \
             metadata_path.open("a", encoding="utf-8") as metadata_out:
            for item in records:
                ids_out.write(json.dumps(item["chunk_id"]) + "\n")
                metadata_out.write(json.dumps({
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "source": item["source"],
                    "language": item["language"],
                    "title": item["title"],
                    "url": item["url"],
                    "chunk_index": item["chunk_index"],
                    "text_hash": item["text_hash"],
                }, ensure_ascii=False) + "\n")

    vectors.flush()
    actual_count = write_index
    if texts:
        actual_count += len(embeddings)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": str(chunks_path),
        "model": args.model,
        "dimension": dimension,
        "count": actual_count,
        "batch_size": args.batch_size,
        "dtype": "float32",
        "normalized": True,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "vectors": str(vectors_path),
        "chunk_ids": str(ids_path),
        "metadata": str(metadata_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
