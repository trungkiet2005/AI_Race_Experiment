"""Fail-closed integrity audit for the public GPU result archives.

The archives are immutable copies from the shared GreenNode volume.  This
script verifies their expected hashes, rejects unsafe tar paths, parses every
JSON metadata file, and emits a compact machine-readable inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED = {
    "analysis-pilot-identified-t1-0.tar.gz": {
        "sha256": "c7c7e8569fb347dae68bba6b5a7f0796a2333ea9a0a064d404fb7f91c2728109",
        "evidence_class": "pilot-analysis",
        "study": "persona-sensitivity",
    },
    "game-understanding-results.tar.gz": {
        "sha256": "7c895cf719b53a36b52c9c895e3c1788369f8d39c8665221f250b7e68f3c6a35",
        "evidence_class": "smoke-and-pilot-raw",
        "study": "game-understanding",
    },
    "pilot-identified-t1-0-results.tar.gz": {
        "sha256": "6ef49b248bae0b1b2fedff575c73bbd5900c9a478a0f41a3d9b3614570a65516",
        "evidence_class": "pilot-raw",
        "study": "persona-sensitivity",
    },
    "smoke-results.tar.gz": {
        "sha256": "2213c3acaa2970c4143d1484ac3a333600883676100b08820124233941f675e8",
        "evidence_class": "smoke-raw",
        "study": "persona-sensitivity",
    },
    "smoke-t1-0-results.tar.gz": {
        "sha256": "546b21a0e46ed37666345dfd469b773224ecd9a42ab9794a0dd868ff91410636",
        "evidence_class": "smoke-raw",
        "study": "persona-sensitivity",
    },
    "smoke-t1-2-results.tar.gz": {
        "sha256": "38dd0f0cd871043fb50cf13449f80076c307dd9a8797f301cca1f534d8f1cde3",
        "evidence_class": "smoke-raw",
        "study": "persona-sensitivity",
    },
    "surface-pilot-analysis-v2.tar.gz": {
        "sha256": "72a84946d438d54c431b7d940a8c22589b40e2415b9e45a2ee091352866d64aa",
        "evidence_class": "pilot-analysis",
        "study": "surface-sensitivity",
    },
    "surface-pilot-raw.tar.gz": {
        "sha256": "366a027900c58ade9e8f564885c3d184bda499d4b0f34d40630f76719c10dc98",
        "evidence_class": "pilot-raw",
        "study": "surface-sensitivity",
    },
    "surface-smoke-analysis-v2.tar.gz": {
        "sha256": "8a7f97dc1f9ea30575d27795e690ecad28562dede2b799c54be2c05d4e4f91a7",
        "evidence_class": "smoke-analysis",
        "study": "surface-sensitivity",
    },
    "surface-smoke-analysis-v1.tar.gz": {
        "sha256": "2e7fb769f6c70b5dac498f3bbab728526a81817b6499e3eb3c7770ec77be4621",
        "evidence_class": "superseded-smoke-analysis",
        "study": "surface-sensitivity",
    },
    "surface-smoke-raw.tar.gz": {
        "sha256": "f96e20f0c75aefff64a69ed0002f1e94fb97e465b6d460eb3705702441ede41e",
        "evidence_class": "smoke-raw",
        "study": "surface-sensitivity",
    },
}

FORBIDDEN_PUBLIC_MARKERS = (
    b"103.73.232.",
    b"root@",
    b"BEGIN OPENSSH PRIVATE KEY",
    b"BEGIN RSA PRIVATE KEY",
    b"gho_",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def audit_archive(path: Path, expected: dict[str, str]) -> dict[str, object]:
    actual_hash = sha256(path)
    if actual_hash != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual_hash}")

    statuses: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    run_manifests = 0
    json_metadata_files = 0
    total_races = 0
    total_turns = 0
    member_names: set[str] = set()
    scanned_files = 0

    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if not safe_member_name(member.name):
                raise ValueError(f"Unsafe member path in {path.name}: {member.name}")
            if member.name in member_names:
                raise ValueError(f"Duplicate member in {path.name}: {member.name}")
            member_names.add(member.name)
            if not member.isfile():
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            data = extracted.read()
            scanned_files += 1
            for marker in FORBIDDEN_PUBLIC_MARKERS:
                if marker in data:
                    raise ValueError(
                        f"Forbidden public marker {marker!r} in "
                        f"{path.name}:{member.name}"
                    )
            if not member.name.endswith(".json"):
                continue
            try:
                payload = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}:{member.name}: {exc}"
                ) from exc
            json_metadata_files += 1
            if not isinstance(payload, dict):
                continue

            schema = payload.get("schema_version")
            if isinstance(schema, str):
                schemas[schema] += 1
            if member.name.endswith("run_manifest.json"):
                run_manifests += 1
                status = payload.get("status", "missing")
                statuses[str(status)] += 1
                if isinstance(payload.get("n_races"), int):
                    total_races += payload["n_races"]
                if isinstance(payload.get("n_turns"), int):
                    total_turns += payload["n_turns"]

    return {
        "file": path.name,
        "sha256": actual_hash,
        "bytes": path.stat().st_size,
        "study": expected["study"],
        "evidence_class": expected["evidence_class"],
        "members": len(members),
        "scanned_files": scanned_files,
        "json_metadata_files": json_metadata_files,
        "run_manifests": run_manifests,
        "run_statuses": dict(sorted(statuses.items())),
        "manifest_races": total_races,
        "manifest_turns": total_turns,
        "schemas": dict(sorted(schemas.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    actual_names = {path.name for path in args.archive_dir.glob("*.tar.gz")}
    expected_names = set(EXPECTED)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(f"Archive set mismatch; missing={missing}, unexpected={unexpected}")

    records = [
        audit_archive(args.archive_dir / name, EXPECTED[name])
        for name in sorted(EXPECTED)
    ]
    ledger = {
        "schema_version": "ai-race-public-gpu-archive-ledger-v1",
        "status": "verified",
        "archive_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "total_scanned_files": sum(int(record["scanned_files"]) for record in records),
        "privacy_boundary": (
            "Result payloads and scientific manifests only; pod access logs, PIDs, "
            "source snapshots, model caches, credentials, and third-party source copies excluded."
        ),
        "archives": records,
    }
    rendered = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
