"""Cross-reference gate for the two manuscripts and the ARR paper.

Each document is built on its own, so a \\ref in one document to a \\label
defined only in another does not resolve and prints as ??. This script reads
every LaTeX source, collects the labels each document defines and the labels
each document cites, and fails when a citation has no definition inside the
same document.

It also reports labels a document defines and never cites, which is how an
orphaned table survives a split.

Run: python scripts/check_paper_crossrefs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS = {
    "AAMAS main paper": [ROOT / "paper" / "main.tex"],
    "AAMAS supplement": [ROOT / "paper" / "supplementary.tex"],
    "ARR paper": [ROOT / "paper" / "acl" / "main.tex"],
}

LABEL = re.compile(r"\\label\{([^}]*)\}")
REF = re.compile(r"\\(?:auto|c|C|name|page|v|eq)?ref\*?\{([^}]*)\}")
COMMENT = re.compile(r"(?<!\\)%.*$")


def strip_comments(text: str) -> str:
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def collect(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    labels: set[str] = set()
    refs: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing source: {path}")
        body = strip_comments(path.read_text(encoding="utf-8"))
        labels |= set(LABEL.findall(body))
        for group in REF.findall(body):
            for name in group.split(","):
                name = name.strip()
                if name:
                    refs[name] = refs.get(name, 0) + 1
    return labels, refs


def main() -> int:
    failures = 0
    everything: dict[str, set[str]] = {}
    for name, paths in DOCUMENTS.items():
        everything[name] = collect(paths)[0]

    for name, paths in DOCUMENTS.items():
        labels, refs = collect(paths)
        dangling = sorted(r for r in refs if r not in labels)
        print(f"{name}: {len(labels)} labels, {len(refs)} distinct references")
        for target in dangling:
            elsewhere = [d for d, ls in everything.items() if d != name and target in ls]
            where = f" (defined in {', '.join(elsewhere)})" if elsewhere else ""
            print(f"  FAIL unresolved \\ref{{{target}}}{where}")
            failures += 1
        orphans = sorted(
            lab for lab in labels
            if lab not in refs
            and not lab.startswith(("sec:", "app:"))
        )
        for lab in orphans:
            print(f"  WARN label never cited: {lab}")
    print()
    if failures:
        print(f"{failures} unresolved cross-reference(s)")
        return 1
    print("no unresolved cross-references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
