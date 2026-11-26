import argparse
import json
import sys
from pathlib import Path

from .engine import RagEngine, load_jsonl


DEFAULT_MANIFESTS = {
    "e5": "embeddings-pilot/e5-small/manifest.json",
    "word2vec": "embeddings-pilot/word2vec/manifest.json",
}


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def add_engine_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding",
        choices=tuple(DEFAULT_MANIFESTS),
        default="e5",
        help="Representacao vetorial: E5 ou Word2Vec Average.",
    )
    parser.add_argument(
        "--manifest",
        "--embedding-manifest",
        help="Sobrescreve o caminho padrao do manifesto.",
    )
    parser.add_argument(
        "--search",
        "--retrieval",
        choices=("cosine_exact", "hnsw"),
        default="hnsw",
        help="Algoritmo de busca vetorial.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ef", type=int, default=200)
    parser.add_argument("--llm", "--ollama-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.2)


def create_engine(args: argparse.Namespace) -> RagEngine:
    manifest = args.manifest or DEFAULT_MANIFESTS[args.embedding]
    return RagEngine(
        embedding_manifest=Path(manifest),
        retrieval=args.search,
        k=args.k,
        ef=args.ef,
        ollama_model=args.llm,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        progress=progress,
    )


def serialize_result(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def save_and_print_json(path: str | None, value: dict) -> None:
    serialized = serialize_result(value)
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        progress(f"JSON salvo em {output}.")
    print(serialized)


def print_answer(result: dict) -> None:
    print(result["answer"])
    print("\nFontes:")
    for citation in result["citations"]:
        print(f"- {citation['title']}: {citation['url']}")
    print("\nDesempenho:")
    for name, value in result["latency_ms"].items():
        print(f"- {name}: {value:.2f} ms")
    print(
        f"- representacao: {result['embedding']}; "
        f"busca: {result['retrieval']}"
    )


def ask_command(args: argparse.Namespace) -> None:
    engine = create_engine(args)
    result = engine.ask(args.question)
    if args.format == "json":
        save_and_print_json(args.output, result)
    else:
        print_answer(result)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialize_result(result), encoding="utf-8")
            progress(f"JSON salvo em {output}.")


def evaluate_command(args: argparse.Namespace) -> None:
    engine = create_engine(args)
    items = load_jsonl(Path(args.dataset))
    if args.limit:
        items = items[:args.limit]
    progress(f"Perguntas a processar: {len(items)}.")
    results = []
    for number, item in enumerate(items, start=1):
        progress(f"[{number}/{len(items)}] Gerando resposta...")
        result = engine.ask(
            item["question"],
            relevant_chunk_ids=item.get("relevant_chunk_ids"),
            ground_truth=item.get("ground_truth"),
        )
        results.append(result)
        timings = result["latency_ms"]
        progress(
            f"[{number}/{len(items)}] Concluída — "
            f"embedding {timings['embedding']:.1f} ms, "
            f"busca {timings['retrieval']:.1f} ms, "
            f"geração {timings['generation']:.1f} ms."
        )

    report = {
        "ollama_model": args.llm,
        "embedding": engine.representation,
        "retrieval": args.search,
        "k": args.k,
        "ef": args.ef,
        "temperature": args.temperature,
        "startup_ms": engine.startup_ms,
        "questions": len(results),
        "results": results,
    }
    save_and_print_json(args.output, report)


def chat_command(args: argparse.Namespace) -> None:
    engine = create_engine(args)
    progress("Digite uma pergunta. Use /sair para encerrar.")
    while True:
        try:
            question = input("Pergunta> ").strip()
        except EOFError:
            print()
            return
        if question.lower() in {"/sair", "/exit", "/quit"}:
            return
        if not question:
            continue
        print_answer(engine.ask(question))
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="paa-rag",
        description="CLI local de busca semantica e respostas RAG.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask_parser = commands.add_parser("ask", help="Responder uma pergunta.")
    add_engine_options(ask_parser)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--format", choices=("text", "json"), default="text")
    ask_parser.add_argument("--output")
    ask_parser.set_defaults(handler=ask_command)

    chat_parser = commands.add_parser(
        "chat",
        help="Fazer varias perguntas sem recarregar modelos e indice.",
    )
    add_engine_options(chat_parser)
    chat_parser.set_defaults(handler=chat_command)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Executar um conjunto de perguntas para avaliacao RAGAS.",
    )
    add_engine_options(evaluate_parser)
    evaluate_parser.add_argument(
        "--dataset",
        "--questions",
        default="evaluation-pilot/questions-pt-rag.jsonl",
    )
    evaluate_parser.add_argument("--limit", type=int, default=5)
    evaluate_parser.add_argument(
        "--output",
        default="evaluation-pilot/rag-pilot-report.json",
    )
    evaluate_parser.set_defaults(handler=evaluate_command)

    args = parser.parse_args(argv)
    args.handler(args)
