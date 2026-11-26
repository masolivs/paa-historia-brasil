import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Freeze corpus artifacts with SHA-256 hashes.")
    parser.add_argument("--corpus", default="corpus-full")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    validation_path = corpus / "validation.json"
    if not validation_path.exists():
        raise SystemExit(f"Missing validation report: {validation_path}")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise SystemExit("Corpus validation failed; artifacts were not frozen.")

    names = ("manifest.json", "chunks.jsonl", "rejected.jsonl", "dedup.sqlite")
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus),
        "validation_report": str(validation_path),
        "files": {},
    }
    for name in names:
        path = corpus / name
        if not path.exists():
            raise SystemExit(f"Missing corpus artifact: {path}")
        frozen["files"][name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    frozen_path = corpus / "frozen.json"
    frozen_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
