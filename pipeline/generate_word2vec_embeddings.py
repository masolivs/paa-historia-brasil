import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec


WORD_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*|\d+", re.UNICODE)


def tokenize(text):
    return [token.lower() for token in WORD_RE.findall(text)]


def chunk_vector(tokens, model):
    vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv
    ]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    vector = np.mean(vectors, axis=0, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec and generate average-word embeddings.",
    )
    parser.add_argument("--chunks", default="corpus-pilot/chunks.jsonl")
    parser.add_argument("--output", default="embeddings-pilot/word2vec")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    sentences = []
    with chunks_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            sentences.append(tokenize(record["text"]))
            if args.limit and len(records) >= args.limit:
                break

    model = Word2Vec(
        sentences=sentences,
        vector_size=args.dimension,
        window=args.window,
        min_count=args.min_count,
        sg=1,
        workers=args.workers,
        epochs=args.epochs,
    )
    model.save(str(output_dir / "word2vec.model"))

    vectors_path = output_dir / "vectors.f32.npy"
    vectors = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype="float32",
        shape=(len(records), args.dimension),
    )
    ids_path = output_dir / "chunk_ids.jsonl"
    metadata_path = output_dir / "metadata.jsonl"

    with ids_path.open("w", encoding="utf-8") as ids_out, \
         metadata_path.open("w", encoding="utf-8") as metadata_out:
        for index, record in enumerate(records):
            vectors[index] = chunk_vector(tokenize(record["text"]), model)
            ids_out.write(json.dumps(record["chunk_id"]) + "\n")
            metadata_out.write(json.dumps({
                "chunk_id": record["chunk_id"],
                "document_id": record["document_id"],
                "source": record["source"],
                "language": record["language"],
                "title": record["title"],
                "url": record["url"],
                "chunk_index": record["chunk_index"],
                "text_hash": record["text_hash"],
            }, ensure_ascii=False) + "\n")

    vectors.flush()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": str(chunks_path),
        "representation": "word2vec_average",
        "model": str(output_dir / "word2vec.model"),
        "dimension": args.dimension,
        "count": len(records),
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "dtype": "float32",
        "normalized": True,
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
