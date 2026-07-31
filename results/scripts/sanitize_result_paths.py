"""Replace private path prefixes in JSON result metadata before publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, old, new) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument("--directory", type=Path, action="append", default=[])
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()

    paths = list(args.file)
    for directory in args.directory:
        paths.extend(directory.rglob("*.json"))
        paths.extend(directory.rglob("*.csv"))
    if not paths:
        parser.error("provide at least one --file or --directory")

    for path in sorted(set(paths)):
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            sanitized = replace_strings(payload, args.old, args.new)
            rendered = json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n"
        else:
            rendered = path.read_text(encoding="utf-8").replace(args.old, args.new)
        path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
