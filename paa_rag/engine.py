import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer


WORD_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*|\d+", re.UNICODE)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def exact_search(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= vectors.shape[0]:
        raise ValueError(f"k must be between 1 and {vectors.shape[0]}")
    scores = np.asarray(vectors) @ query
    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


class RagEngine:
    def __init__(
        self,
        embedding_manifest: Path,
        retrieval: str,
        k: int,
        ef: int,
        ollama_model: str,
        ollama_url: str,
        temperature: float,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        startup_start = time.perf_counter()
        self.progress = progress or (lambda _: None)
        self.embedding_dir = embedding_manifest.parent
        self.manifest = json.loads(
            embedding_manifest.read_text(encoding="utf-8")
        )
        self.retrieval = retrieval
        self.k = k
        self.ef = ef
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.representation = self.manifest.get(
            "representation",
            "sentence_embedding",
        )

        self.progress("Carregando vetores, IDs e metadados...")
        self.vectors = np.load(
            self.embedding_dir / "vectors.f32.npy",
            mmap_mode="r",
        )
        self.chunk_ids = load_jsonl(self.embedding_dir / "chunk_ids.jsonl")
        self.metadata = load_jsonl(self.embedding_dir / "metadata.jsonl")
        if (
            len(self.chunk_ids) != len(self.metadata)
            or len(self.chunk_ids) != self.vectors.shape[0]
        ):
            raise ValueError("Vector, ID and metadata lengths do not match")
        if not 1 <= k <= self.vectors.shape[0]:
            raise ValueError(
                f"k must be between 1 and {self.vectors.shape[0]}"
            )
        self.metadata_by_id = {
            item["chunk_id"]: item for item in self.metadata
        }
        self.chunk_id_set = set(self.chunk_ids)
        self.progress(f"Carregados {len(self.chunk_ids):,} vetores.")

        chunks_path = Path(self.manifest["chunks"])
        if not chunks_path.is_absolute() and not chunks_path.exists():
            chunks_path = self.embedding_dir / chunks_path
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        self.progress(f"Lendo textos do piloto em {chunks_path}...")
        self.text_by_id = {}
        with chunks_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_id = record["chunk_id"]
                if chunk_id in self.chunk_id_set:
                    self.text_by_id[chunk_id] = record["text"]
        missing = self.chunk_id_set.difference(self.text_by_id)
        if missing:
            raise ValueError(
                f"{len(missing)} embedding IDs have no matching chunk text"
            )
        self.progress(f"Textos carregados: {len(self.text_by_id):,}.")

        if self.representation == "word2vec_average":
            from gensim.models import Word2Vec

            model_path = self.embedding_dir / "word2vec.model"
            self.progress(f"Carregando modelo Word2Vec em {model_path}...")
            self.query_model = Word2Vec.load(str(model_path))
        else:
            model_name = self.manifest["model"]
            self.progress(f"Carregando modelo de embeddings {model_name}...")
            self.query_model = SentenceTransformer(model_name)
        self.progress("Modelo de embeddings carregado.")

        self.index = None
        if self.retrieval == "hnsw":
            self.progress("Construindo índice HNSW...")
            self.index = hnswlib.Index(
                space="cosine",
                dim=self.vectors.shape[1],
            )
            self.index.init_index(
                max_elements=self.vectors.shape[0],
                ef_construction=200,
                M=32,
            )
            self.index.add_items(
                np.asarray(self.vectors),
                np.arange(self.vectors.shape[0]),
            )
            self.index.set_ef(ef)
            self.progress("Índice HNSW pronto.")
        self.startup_ms = (time.perf_counter() - startup_start) * 1000

    def encode_query(self, question: str) -> np.ndarray:
        if self.representation == "word2vec_average":
            tokens = [
                token.lower()
                for token in WORD_RE.findall(question)
            ]
            known = [
                self.query_model.wv[token]
                for token in tokens
                if token in self.query_model.wv
            ]
            if not known:
                raise ValueError(
                    "Nenhuma palavra da pergunta existe no vocabulário Word2Vec"
                )
            vector = np.mean(known, axis=0, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
            return np.asarray(vector, dtype=np.float32)

        query_prefix = self.manifest.get("query_prefix", "query: ")
        return self.query_model.encode(
            [f"{query_prefix}{question}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

    def retrieve(
        self,
        query: np.ndarray,
        k: int | None = None,
    ) -> tuple[np.ndarray, list[float], float]:
        effective_k = min(k or self.k, self.vectors.shape[0])
        if effective_k < 1:
            raise ValueError("k deve ser maior ou igual a 1")

        start = time.perf_counter()
        if self.retrieval == "hnsw":
            if effective_k > self.ef:
                self.index.set_ef(effective_k)
            indices, distances = self.index.knn_query(query, k=effective_k)
            result = indices[0].astype(np.int64)
            scores = [float(1.0 - dist) for dist in distances[0]]
        else:
            result = exact_search(self.vectors, query, effective_k)
            scores = [float(np.dot(self.vectors[idx], query)) for idx in result]
        latency_ms = (time.perf_counter() - start) * 1000
        return result, scores, latency_ms

    def generate(self, prompt: str) -> dict:
        payload = json.dumps({
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    def ask(
        self,
        question: str,
        relevant_chunk_ids: list[str] | None = None,
        ground_truth: str | None = None,
        k: int | None = None,
    ) -> dict:
        self.progress("Gerando embedding da pergunta...")
        start = time.perf_counter()
        query = self.encode_query(question)
        embedding_ms = (time.perf_counter() - start) * 1000

        self.progress("Buscando chunks...")
        indices, scores, retrieval_ms = self.retrieve(query, k=k)
        retrieved_ids = [self.chunk_ids[index] for index in indices]
        contexts = [self.text_by_id[chunk_id] for chunk_id in retrieved_ids]
        citations = []
        for chunk_id, score in zip(retrieved_ids, scores):
            metadata = self.metadata_by_id[chunk_id]
            citations.append({
                "chunk_id": chunk_id,
                "title": metadata.get("title"),
                "source": metadata.get("source"),
                "url": metadata.get("url"),
                "score": round(score, 4),
            })

        context_blocks = []
        for position, (citation, text) in enumerate(zip(citations, contexts), start=1):
            title = citation.get("title") or "Documento sem título"
            url = citation.get("url") or "URL não informada"
            context_blocks.append(
                f"[Fonte {position}] Título: {title}\nURL: {url}\nTrecho: {text}"
            )
        context_text = "\n\n".join(context_blocks)

        prompt = (
            "Responda em português usando apenas os contextos abaixo. "
            "Cite explicitamente as fontes no texto (ex.: [Fonte 1]). "
            "Se a resposta não estiver nos contextos, diga claramente que não encontrou. "
            "Ao final, liste apenas as citações usadas com Título e URL exatamente como fornecidos, "
            "sem inventar links ou títulos.\n\n"
            f"Contextos:\n{context_text}\n\n"
            f"Pergunta: {question}\n"
            "Resposta:"
        )

        self.progress(
            f"Enviando contexto à LLM local ({self.ollama_model})..."
        )
        generation_start = time.perf_counter()
        response = self.generate(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000
        self.progress("Geração da resposta concluída.")
        result = {
            "question": question,
            "answer": response.get("response", "").strip(),
            "ollama_model": self.ollama_model,
            "contexts": contexts,
            "retrieved_chunk_ids": retrieved_ids,
            "citations": citations,
            "embedding": self.representation,
            "retrieval": self.retrieval,
            "startup_ms": self.startup_ms,
            "latency_ms": {
                "embedding": embedding_ms,
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": embedding_ms + retrieval_ms + generation_ms,
            },
        }
        if relevant_chunk_ids is not None:
            result["relevant_chunk_ids"] = relevant_chunk_ids
        if ground_truth is not None:
            result["ground_truth"] = ground_truth
        return result