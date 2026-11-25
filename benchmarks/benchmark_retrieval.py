import argparse
import json
import time
from pathlib import Path

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer


WORD_RE = None


def encode_questions(manifest, embedding_dir, questions):
    global WORD_RE
    if manifest.get("representation") == "word2vec_average":
        import re
        from gensim.models import Word2Vec
        WORD_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*|\d+", re.UNICODE)
        model = Word2Vec.load(str(embedding_dir / "word2vec.model"))
        vectors = []
        for item in questions:
            tokens = [token.lower() for token in WORD_RE.findall(item["question"])]
            known = [model.wv[token] for token in tokens if token in model.wv]
            if known:
                vector = np.mean(known, axis=0, dtype=np.float32)
                norm = np.linalg.norm(vector)
                if norm:
                    vector /= norm
            else:
                vector = np.zeros(model.vector_size, dtype=np.float32)
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)

    model = SentenceTransformer(manifest["model"])
    return model.encode(
        ["query: " + item["question"] for item in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def average_precision(ranks, relevant):
    if not relevant:
        return 0.0
    found = 0
    score = 0.0
    for position, chunk_id in enumerate(ranks, 1):
        if chunk_id in relevant:
            found += 1
            score += found / position
    return score / len(relevant)


def reciprocal_rank(ranks, relevant):
    for position, chunk_id in enumerate(ranks, 1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def recall_at_k(ranks, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranks[:k]).intersection(relevant)) / len(relevant)


def load_ids(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark exact cosine and HNSW retrieval.")
    parser.add_argument("--embedding-manifest", default="embeddings-pilot/e5-small/manifest.json")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--ef", type=int, default=200)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--output", default="embeddings-pilot/e5-small/retrieval-report.json")
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_manifest).parent
    manifest = json.loads(Path(args.embedding_manifest).read_text(encoding="utf-8"))
    vectors = np.load(embedding_dir / "vectors.f32.npy", mmap_mode="r")
    chunk_ids = load_ids(embedding_dir / "chunk_ids.jsonl")
    if len(chunk_ids) != vectors.shape[0]:
        raise SystemExit("vectors.f32.npy and chunk_ids.jsonl have different lengths")

    questions = []
    with Path(args.questions).open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                questions.append(json.loads(line))
    if not questions:
        raise SystemExit("Question file is empty")

    query_vectors = encode_questions(manifest, embedding_dir, questions)

    hnsw = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    hnsw.init_index(max_elements=vectors.shape[0], ef_construction=200, M=args.m)
    hnsw.add_items(np.asarray(vectors), np.arange(vectors.shape[0]))
    hnsw.set_ef(args.ef)

    metrics = {
        "cosine_exact": {"latencies_ms": [], "reciprocal_ranks": [], "average_precisions": []},
        "hnsw": {"latencies_ms": [], "reciprocal_ranks": [], "average_precisions": []},
    }
    for k in args.k:
        for method in metrics.values():
            method[f"recall@{k}"] = []

    for query_index, query in enumerate(query_vectors):
        relevant = set(questions[query_index]["relevant_chunk_ids"])
        max_k = max(args.k)

        start = time.perf_counter()
        scores = np.asarray(vectors) @ query
        exact_indices = np.argpartition(scores, -max_k)[-max_k:]
        exact_indices = exact_indices[np.argsort(scores[exact_indices])[::-1]]
        exact_ms = (time.perf_counter() - start) * 1000
        exact_ranks = [chunk_ids[index] for index in exact_indices]

        start = time.perf_counter()
        hnsw_indices, _ = hnsw.knn_query(query, k=max_k)
        hnsw_ms = (time.perf_counter() - start) * 1000
        hnsw_indices = np.asarray(hnsw_indices[0], dtype=np.int64)
        hnsw_ranks = [chunk_ids[index] for index in hnsw_indices]

        for name, ranks, latency in (
            ("cosine_exact", exact_ranks, exact_ms),
            ("hnsw", hnsw_ranks, hnsw_ms),
        ):
            method = metrics[name]
            method["latencies_ms"].append(latency)
            method["reciprocal_ranks"].append(reciprocal_rank(ranks, relevant))
            method["average_precisions"].append(average_precision(ranks, relevant))
            for k in args.k:
                method[f"recall@{k}"].append(recall_at_k(ranks, relevant, k))

    report = {"questions": len(questions), "k": args.k, "hnsw": {"ef": args.ef, "M": args.m}}
    for name, method in metrics.items():
        report[name] = {
            "latency_ms_mean": float(np.mean(method["latencies_ms"])),
            "latency_ms_p95": float(np.percentile(method["latencies_ms"], 95)),
            "mrr": float(np.mean(method["reciprocal_ranks"])),
            "map": float(np.mean(method["average_precisions"])),
        }
        for k in args.k:
            report[name][f"recall@{k}_mean"] = float(np.mean(method[f"recall@{k}"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
