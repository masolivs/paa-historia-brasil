import argparse
import datetime
import hashlib
import html
import json
import re
import unicodedata
import sqlite3
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()


SOURCES = {
    "pt": ("extracted-pt", "ptwiki"),
    "es": ("extracted-es", "eswiki"),
    "en": ("extracted-en-brazil", "enwiki"),
}

REDIRECT_PREFIXES = (
    "#REDIRECIONAMENTO",
    "#REDIRECCION",
    "#REDIRECCIÓN",
    "#REDIRECT",
)


def clean_text(text):
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    template_pattern = re.compile(r"\{\{([^{}]*)\}\}")
    while True:
        cleaned = template_pattern.sub(clean_template, text)
        if cleaned == text:
            break
        text = cleaned

    text = re.sub(r"\[\[([^|\[\]]+)\|([^|\[\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^|\[\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("{{", " ").replace("}}", " ")
    text = text.replace("[[", " ").replace("]]", " ")
    text = text.replace("|", " ")
    for redirect in REDIRECT_PREFIXES:
        text = text.replace(redirect, " ")

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    lines = []
    previous_blank = False
    for line in text.split("\n"):
        line = " ".join(line.split())
        blank = not line
        if blank:
            if not previous_blank and lines:
                lines.append("")
        else:
            lines.append(line)
        previous_blank = blank

    return "\n".join(lines).strip()


def clean_template(match):
    body = match.group(1)
    name, *parameters = body.split("|")
    name = name.strip().lower()
    if name in {"fmtn", "formatnum", "nts"} and parameters:
        return parameters[-1].strip()
    return ""


def is_invalid(title, text):
    title_lower = (title or "").lower()
    text_upper = (text or "").lstrip().upper()

    if not text or len(text) < 100:
        return True
    if any(text_upper.startswith(prefix) for prefix in REDIRECT_PREFIXES):
        return True
    if "desambig" in title_lower:
        return True
    return False


def approximate_chunks(text, target_chars=2400, overlap_chars=300):
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else current + "\n\n" + paragraph
        if len(candidate) <= target_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = current[-overlap_chars:].strip()

        remaining = (current + "\n\n" + paragraph).strip() if current else paragraph
        while len(remaining) > target_chars:
            cut = remaining.rfind(". ", 0, target_chars)
            if cut < target_chars // 2:
                cut = remaining.rfind(" ", 0, target_chars)
            if cut < target_chars // 2:
                cut = target_chars

            cut_chunk = remaining[:cut].strip()
            if cut_chunk:
                chunks.append(cut_chunk)
            start = max(0, cut - overlap_chars)
            remaining = remaining[start:].strip()

        current = remaining

    if current:
        chunks.append(current)

    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 100]


def chunk_by_tokens(text, tokenizer, max_tokens=512, overlap_tokens=64):
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks = []
    current_ids = []
    content_limit = max(1, max_tokens - 32)

    def encode(unit):
        return tokenizer.encode(unit, add_special_tokens=False)

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

    for paragraph in paragraphs:
        paragraph_ids = encode(paragraph)
        if current_ids:
            candidate = current_ids + paragraph_ids
            if len(candidate) <= content_limit:
                current_ids = candidate
                continue
            chunks.append(decode(current_ids))
            current_ids = current_ids[-overlap_tokens:] if overlap_tokens else []

        if len(paragraph_ids) <= content_limit:
            candidate = current_ids + paragraph_ids
            if len(candidate) <= content_limit:
                current_ids = candidate
            else:
                chunks.append(decode(current_ids))
                current_ids = paragraph_ids
            continue

        step = max(1, content_limit - overlap_tokens)
        for start in range(0, len(paragraph_ids), step):
            ids = paragraph_ids[start:start + content_limit]
            chunk = decode(ids)
            if chunk:
                chunks.append(chunk)
        current_ids = []

    if current_ids:
        chunks.append(decode(current_ids))

    return [chunk for chunk in chunks if len(chunk) >= 100]


def iter_documents(base_dir):
    base = Path(base_dir)
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    yield {
                        "_invalid_json": True,
                        "_path": str(path),
                        "_line": line_number,
                        "_error": str(exc),
                    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    manifest_path = output_dir / "manifest.json"
    rejected_path = output_dir / "rejected.jsonl"

    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_chars_per_chunk": args.chunk_chars,
        "overlap_chars": args.overlap_chars,
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "chunking_mode": "tokens" if args.tokenizer else "characters",
        "sources": {},
    }
    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    global_chunks = 0
    limit_reached = False
    dedup_db = sqlite3.connect(output_dir / "dedup.sqlite")
    dedup_db.execute(
        "CREATE TABLE IF NOT EXISTS seen_text (text_hash TEXT PRIMARY KEY)"
    )
    dedup_db.execute(
        "CREATE TABLE IF NOT EXISTS seen_chunk (text_hash TEXT PRIMARY KEY)"
    )
    dedup_db.execute(
        "CREATE TABLE IF NOT EXISTS seen_chunk_id (chunk_id TEXT PRIMARY KEY)"
    )
    dedup_db.execute("PRAGMA journal_mode=WAL")
    dedup_db.execute("PRAGMA synchronous=NORMAL")

    with chunks_path.open("w", encoding="utf-8") as chunks_out, \
         rejected_path.open("w", encoding="utf-8") as rejected_out:
        for language, (directory, source_name) in SOURCES.items():
            if limit_reached:
                break

            stats = {
                "directory": directory,
                "source": source_name,
                "documents_read": 0,
                "documents_accepted": 0,
                "documents_rejected": 0,
                "invalid_json": 0,
                "duplicate_text": 0,
                "duplicate_chunk": 0,
                "chunks": 0,
                "text_bytes_utf8": 0,
            }

            for document in iter_documents(directory):
                if isinstance(document, dict) and document.get("_invalid_json") is True:
                    stats["invalid_json"] += 1
                    rejected_out.write(json.dumps(document, ensure_ascii=False) + "\n")
                    continue

                stats["documents_read"] += 1
                title = str(document.get("title") or "")
                text = clean_text(document.get("text") or "")

                if is_invalid(title, text):
                    stats["documents_rejected"] += 1
                    rejected_out.write(json.dumps({
                        "source": source_name,
                        "id": document.get("id"),
                        "revid": document.get("revid"),
                        "title": title,
                        "reason": "empty_too_short_or_redirect",
                    }, ensure_ascii=False) + "\n")
                    continue

                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                try:
                    dedup_db.execute(
                        "INSERT INTO seen_text(text_hash) VALUES (?)",
                        (text_hash,),
                    )
                except sqlite3.IntegrityError:
                    stats["duplicate_text"] += 1
                    continue

                stats["documents_accepted"] += 1
                stats["text_bytes_utf8"] += len(text.encode("utf-8"))

                document_id = str(document.get("id") or "")
                revision_id = str(document.get("revid") or "")
                if tokenizer is not None:
                    chunks = chunk_by_tokens(
                        text,
                        tokenizer,
                        max_tokens=args.max_tokens,
                        overlap_tokens=args.overlap_tokens,
                    )
                else:
                    chunks = approximate_chunks(
                        text,
                        target_chars=args.chunk_chars,
                        overlap_chars=args.overlap_chars,
                    )

                for chunk_index, chunk in enumerate(chunks):
                    chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    try:
                        dedup_db.execute(
                            "INSERT INTO seen_chunk(text_hash) VALUES (?)",
                            (chunk_hash,),
                        )
                    except sqlite3.IntegrityError:
                        stats["duplicate_chunk"] += 1
                        continue
                    base_chunk_id = f"{source_name}:{document_id}:{revision_id}:{chunk_index}"
                    chunk_id = base_chunk_id
                    occurrence = 1
                    while True:
                        try:
                            dedup_db.execute(
                                "INSERT INTO seen_chunk_id(chunk_id) VALUES (?)",
                                (chunk_id,),
                            )
                            break
                        except sqlite3.IntegrityError:
                            occurrence += 1
                            chunk_id = f"{base_chunk_id}:{occurrence}"
                    record = {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "revision_id": revision_id,
                        "source": source_name,
                        "language": language,
                        "title": title,
                        "url": document.get("url"),
                        "chunk_index": chunk_index,
                        "text_hash": chunk_hash,
                        "character_count": len(chunk),
                        "token_count": (
                            len(tokenizer.encode(chunk, add_special_tokens=False))
                            if tokenizer is not None
                            else None
                        ),
                        "text": chunk,
                    }
                    chunks_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["chunks"] += 1
                    global_chunks += 1
                    if global_chunks % 10000 == 0:
                        dedup_db.commit()

                    if args.limit_chunks and global_chunks >= args.limit_chunks:
                        limit_reached = True
                        break

                if args.limit_documents and stats["documents_accepted"] >= args.limit_documents:
                    break
                if args.limit_chunks_per_language and stats["chunks"] >= args.limit_chunks_per_language:
                    break
                if limit_reached:
                    break

            stats["text_gib"] = round(stats["text_bytes_utf8"] / (1024 ** 3), 6)
            manifest["sources"][language] = stats

    manifest["chunks_file"] = str(chunks_path)
    manifest["rejected_file"] = str(rejected_path)
    manifest["unique_documents_text"] = sum(
        stats["documents_accepted"]
        for stats in manifest["sources"].values()
    )
    manifest["total_chunks"] = sum(
        stats["chunks"]
        for stats in manifest["sources"].values()
    )
    manifest["total_text_bytes_utf8"] = sum(
        stats["text_bytes_utf8"]
        for stats in manifest["sources"].values()
    )
    manifest["total_text_gib"] = round(
        manifest["total_text_bytes_utf8"] / (1024 ** 3),
        6,
    )
    dedup_db.close()

    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Clean extracted Wikipedia JSONL and generate a chunked corpus.",
    )
    parser.add_argument("--output", default="corpus")
    parser.add_argument("--chunk-chars", type=int, default=2400)
    parser.add_argument("--overlap-chars", type=int, default=300)
    parser.add_argument("--limit-documents", type=int, default=0)
    parser.add_argument("--limit-chunks", type=int, default=0)
    parser.add_argument("--limit-chunks-per-language", type=int, default=0)
    parser.add_argument("--tokenizer", default="intfloat/multilingual-e5-small")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
