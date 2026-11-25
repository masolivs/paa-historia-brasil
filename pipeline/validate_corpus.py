import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()


REQUIRED_FIELDS = (
    "chunk_id",
    "document_id",
    "revision_id",
    "source",
    "language",
    "title",
    "url",
    "chunk_index",
    "text_hash",
    "character_count",
    "token_count",
    "text",
)


def validate(args):
    random.seed(args.seed)
    chunks_path = Path(args.chunks)
    rejected_path = Path(args.rejected)
    manifest_path = chunks_path.parent / "manifest.json"
    validation_db_path = chunks_path.parent / "validation.sqlite"
    chunking_mode = "characters"
    manifest = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as manifest:
            manifest = json.load(manifest)
            chunking_mode = manifest.get("chunking_mode", "characters")

    stats = {
        "chunks": 0,
        "invalid_json": 0,
        "missing_fields": 0,
        "wrong_character_count": 0,
        "wrong_text_hash": 0,
        "duplicate_chunk_id": 0,
        "duplicate_text_hash": 0,
        "over_chunk_limit": 0,
        "over_token_limit": 0,
        "wrong_token_count": 0,
        "suspicious_wikitext": 0,
        "languages": Counter(),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    samples = {language: [] for language in ("pt", "es", "en")}
    seen_per_language = Counter()
    validation_db_path.unlink(missing_ok=True)
    validation_db = sqlite3.connect(validation_db_path)
    validation_db.execute("CREATE TABLE seen_chunk_id (value TEXT PRIMARY KEY)")
    validation_db.execute("CREATE TABLE seen_text_hash (value TEXT PRIMARY KEY)")
    validation_db.execute("PRAGMA journal_mode=OFF")
    validation_db.execute("PRAGMA synchronous=OFF")
    validation_db.execute("PRAGMA temp_store=FILE")

    with chunks_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            stats["chunks"] += 1
            if any(field not in record for field in REQUIRED_FIELDS):
                stats["missing_fields"] += 1

            text = record.get("text") or ""
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = record.get("chunk_id")
            record_hash = record.get("text_hash")

            try:
                validation_db.execute(
                    "INSERT INTO seen_chunk_id(value) VALUES (?)",
                    (chunk_id,),
                )
            except sqlite3.IntegrityError:
                stats["duplicate_chunk_id"] += 1

            try:
                validation_db.execute(
                    "INSERT INTO seen_text_hash(value) VALUES (?)",
                    (record_hash,),
                )
            except sqlite3.IntegrityError:
                stats["duplicate_text_hash"] += 1

            if record.get("character_count") != len(text):
                stats["wrong_character_count"] += 1
            if record_hash != text_hash:
                stats["wrong_text_hash"] += 1
            if len(text) > args.max_characters:
                stats["over_chunk_limit"] += 1
            actual_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if actual_tokens > args.max_tokens:
                stats["over_token_limit"] += 1
            if record.get("token_count") != actual_tokens:
                stats["wrong_token_count"] += 1
            if "{{" in text or "}}" in text or "[[Category:" in text:
                stats["suspicious_wikitext"] += 1

            language = record.get("language")
            stats["languages"][language] += 1

            if language in samples:
                seen_per_language[language] += 1
                sample = samples[language]
                if len(sample) < args.sample_size:
                    sample.append(record)
                else:
                    replacement = random.randrange(seen_per_language[language])
                    if replacement < args.sample_size:
                        sample[replacement] = record
            if stats["chunks"] % 100000 == 0:
                validation_db.commit()

    validation_db.commit()
    validation_db.close()
    validation_db_path.unlink(missing_ok=True)

    rejected = 0
    invalid_rejections = 0
    if rejected_path.exists():
        with rejected_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                rejected += 1
                try:
                    record = json.loads(line)
                    if "reason" not in record:
                        invalid_rejections += 1
                except json.JSONDecodeError:
                    invalid_rejections += 1

    manifest_chunks_match = (
        manifest is not None
        and manifest.get("total_chunks") == stats["chunks"]
    )
    manifest_text_bytes_match = (
        manifest is not None
        and manifest.get("total_text_bytes_utf8")
        == sum(
            source.get("text_bytes_utf8", 0)
            for source in manifest.get("sources", {}).values()
        )
    )

    report = {
        "chunks_file": str(chunks_path),
        "rejected_file": str(rejected_path),
        "manifest_file": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_chunks_match": manifest_chunks_match,
        "manifest_text_bytes_match": manifest_text_bytes_match,
        "rejected_records": rejected,
        "invalid_rejection_records": invalid_rejections,
        "samples_per_language": args.sample_size,
        "random_seed": args.seed,
        "max_characters": args.max_characters,
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        **stats,
        "languages": dict(stats["languages"]),
        "sample_titles": {
            language: [record["title"] for record in records]
            for language, records in samples.items()
        },
        "passed": (
            stats["invalid_json"] == 0
            and stats["missing_fields"] == 0
            and stats["wrong_character_count"] == 0
            and stats["wrong_text_hash"] == 0
            and stats["duplicate_chunk_id"] == 0
            and stats["duplicate_text_hash"] == 0
            and (chunking_mode == "tokens" or stats["over_chunk_limit"] == 0)
            and stats["over_token_limit"] == 0
            and stats["wrong_token_count"] == 0
            and stats["suspicious_wikitext"] == 0
            and invalid_rejections == 0
            and manifest_path.exists()
            and manifest_chunks_match
            and manifest_text_bytes_match
        ),
    }
    output_path = chunks_path.parent / "validation.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Validate a cleaned and chunked corpus.")
    parser.add_argument("--chunks", default="corpus-pilot/chunks.jsonl")
    parser.add_argument("--rejected", default="corpus-pilot/rejected.jsonl")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--max-characters", type=int, default=2400)
    parser.add_argument("--tokenizer", default="intfloat/multilingual-e5-small")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
