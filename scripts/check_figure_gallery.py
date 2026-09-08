#!/usr/bin/env python3
"""Validate the canonical figure hub and every manuscript figure reference."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MANIFEST = FIGURES / "manifest.json"
VISUAL_EXTENSIONS = (".pdf", ".png", ".svg", ".eps", ".jpg", ".jpeg")
INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_visual(base: Path, reference: str) -> Path | None:
    candidate = (base / reference).resolve()
    options = [candidate]
    if candidate.suffix == "":
        options.extend(candidate.with_suffix(extension) for extension in VISUAL_EXTENSIONS)
    for option in options:
        if option.is_file() and ROOT in option.parents:
            return option
    return None


def check_manifest(errors: list[str]) -> None:
    if not MANIFEST.is_file():
        errors.append("missing figures/manifest.json")
        return
    try:
        records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read figures/manifest.json: {exc}")
        return

    canonical_paths: set[str] = set()
    hashes: set[str] = set()
    for record in records:
        canonical_name = record.get("canonical_path", "")
        canonical = ROOT / canonical_name
        if canonical_name in canonical_paths:
            errors.append(f"duplicate canonical path: {canonical_name}")
        canonical_paths.add(canonical_name)
        if not canonical.is_file():
            errors.append(f"missing canonical figure: {canonical_name}")
            continue
        expected = record.get("sha256", "")
        actual = digest(canonical)
        if expected != actual:
            errors.append(f"hash mismatch: {canonical_name}")
        if actual in hashes:
            errors.append(f"duplicate hash admitted as a separate record: {canonical_name}")
        hashes.add(actual)
        for source_name in record.get("sources", []):
            if not (ROOT / source_name).is_file():
                errors.append(f"missing figure source: {source_name}")


def check_latex_references(errors: list[str]) -> None:
    documents = ((ROOT / "paper", "main.tex"), (ROOT / "paper", "supplementary.tex"), (ROOT / "slides", "ai_race_research_deck.tex"))
    for base, filename in documents:
        path = base / filename
        text = path.read_text(encoding="utf-8", errors="replace")
        for reference in INCLUDEGRAPHICS.findall(text):
            if resolve_visual(base, reference) is None and resolve_visual(ROOT, reference) is None:
                errors.append(f"unresolved figure in {path.relative_to(ROOT)}: {reference}")


def check_html_references(errors: list[str]) -> None:
    for relative in ("web/index.html", "slides/index.html"):
        path = ROOT / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        for reference in re.findall(r"(?:src|href)=\"([^\"]*figures/paper/[^\"]+)\"", text):
            if resolve_visual(path.parent, reference) is None:
                errors.append(f"unresolved visual in {relative}: {reference}")


def main() -> int:
    errors: list[str] = []
    check_manifest(errors)
    check_latex_references(errors)
    check_html_references(errors)
    for relative in ("README.md", "CLAUDE.md", "paper", "slides", "web", "scripts", "figures"):
        target = ROOT / relative
        paths = [target] if target.is_file() else target.rglob("*") if target.is_dir() else []
        for path in paths:
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path == Path(__file__).resolve()
                or path.suffix.lower() in VISUAL_EXTENSIONS
                or path.name == "manifest.json"
            ):
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            if "paper/figures" in content or "results/artifacts/figure_gallery" in content:
                errors.append(f"stale figure path in {path.relative_to(ROOT)}")
    if errors:
        print("figure gallery QA failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("figure gallery QA passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
