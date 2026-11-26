import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.schemas import AnswerResponse, Citation, QuestionRequest, Timings

load_dotenv()

RAG_MANIFEST = Path(os.getenv("RAG_MANIFEST", "embeddings-pilot/e5-small/manifest.json"))
RAG_RETRIEVAL = os.getenv("RAG_RETRIEVAL", "hnsw")
RAG_EF = int(os.getenv("RAG_EF", "200"))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))

_engine: Any = None
_engine_lock = threading.Lock()


def _safe_url(raw_url: Any) -> str | None:
    if not raw_url:
        return None
    text = str(raw_url).strip()
    if text.startswith(("http://", "https://")):
        return text
    return None


def get_engine() -> Any:
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        if not RAG_MANIFEST.is_file():
            raise FileNotFoundError(
                f"Manifesto de embeddings não encontrado em '{RAG_MANIFEST}'. "
                "Gere os embeddings do piloto antes de consultar a API."
            )

        from paa_rag.engine import RagEngine

        _engine = RagEngine(
            embedding_manifest=RAG_MANIFEST,
            retrieval=RAG_RETRIEVAL,
            k=settings.top_k_results,
            ef=RAG_EF,
            ollama_model=settings.ollama_model,
            ollama_url=settings.ollama_base_url,
            temperature=RAG_TEMPERATURE,
            progress=lambda msg: print(f"[paa_rag] {msg}", flush=True),
        )
        return _engine


app = FastAPI(
    title="PAA — Busca Semântica e RAG Local",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "mock_mode": settings.mock_mode,
        "model": settings.ollama_model,
        "retrieval": RAG_RETRIEVAL,
        "manifest": str(RAG_MANIFEST),
        "engine_loaded": _engine is not None,
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest) -> AnswerResponse:
    total_start = time.perf_counter()
    top_k = payload.top_k or settings.top_k_results

    if settings.mock_mode:
        mock_citations = [
            Citation(
                document="Proclamação da República do Brasil",
                source="Amostra Mock (Wikipedia PT)",
                source_url="https://pt.wikipedia.org/wiki/Proclama%C3%A7%C3%A3o_da_Rep%C3%BAblica_do_Brasil",
                doc_id="mock-republica",
                snippet="A Proclamação da República do Brasil foi um levante político-militar ocorrido em 15 de novembro de 1889 que instaurou a forma republicana federativa presidencialista.",
                score=0.92,
            ),
            Citation(
                document="Lei Áurea",
                source="Amostra Mock (Wikipedia PT)",
                source_url="https://pt.wikipedia.org/wiki/Lei_%C3%81urea",
                doc_id="mock-lei-aurea",
                snippet="A Lei Áurea, sancionada em 13 de maio de 1888 pela Princesa Isabel, extinguiu oficialmente a escravidão no Brasil.",
                score=0.78,
            ),
        ]
        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        return AnswerResponse(
            answer=(
                f"[MODO MOCK] Resposta simulada para: '{payload.question}'. "
                "A Proclamação da República ocorreu em 15 de novembro de 1889 [Fonte 1], "
                "pouco após a abolição formal da escravidão pela Lei Áurea em 1888 [Fonte 2]."
            ),
            citations=mock_citations[:top_k],
            timings=Timings(
                embedding_ms=0.0,
                search_ms=0.0,
                llm_ms=0.0,
                total_ms=total_ms,
            ),
            model=f"{settings.ollama_model} (mock)",
        )

    try:
        engine = await asyncio.to_thread(get_engine)
        result = await asyncio.to_thread(engine.ask, payload.question, None, None, top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar pergunta: {exc}",
        ) from exc

    citations: list[Citation] = []
    for meta, snippet in zip(result.get("citations", []), result.get("contexts", [])):
        citations.append(
            Citation(
                document=str(meta.get("title") or "Documento sem título"),
                source=str(meta.get("source") or "Wikipédia"),
                source_url=_safe_url(meta.get("url")),
                doc_id=str(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
                snippet=str(snippet),
                score=round(float(meta.get("score", 0.0)), 4),
            )
        )

    latency = result.get("latency_ms", {})
    embedding_ms = round(float(latency.get("embedding", 0.0)), 2)
    search_ms = round(float(latency.get("retrieval", 0.0)), 2)
    llm_ms = round(float(latency.get("generation", 0.0)), 2)
    total_ms = round(float(latency.get("total", (time.perf_counter() - total_start) * 1000)), 2)

    return AnswerResponse(
        answer=str(result.get("answer") or "").strip(),
        citations=citations,
        timings=Timings(
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        ),
        model=str(result.get("ollama_model") or settings.ollama_model),
    )