"""Compile the paper and deck into the canonical ``results/`` artifact tree.

LaTeX auxiliary files are isolated under ``results/_build/latex/current`` so a
repository build never recreates the retired top-level ``output/`` tree.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "results" / "_build" / "latex" / "current"
PUBLICATION_DIR = ROOT / "results" / "artifacts" / "publication"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def latex(source: str, *, jobname: str) -> None:
    run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={BUILD_DIR.as_posix()}",
            f"-jobname={jobname}",
            source,
        ]
    )


def build_paper() -> Path:
    latex("paper/main.tex", jobname="ai_race_paper")
    run(["bibtex", (BUILD_DIR / "ai_race_paper").as_posix()])
    latex("paper/main.tex", jobname="ai_race_paper")
    latex("paper/main.tex", jobname="ai_race_paper")
    return BUILD_DIR / "ai_race_paper.pdf"


def build_deck() -> Path:
    latex("slides/ai_race_research_deck.tex", jobname="ai_race_research_deck")
    latex("slides/ai_race_research_deck.tex", jobname="ai_race_research_deck")
    return BUILD_DIR / "ai_race_research_deck.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper-only", action="store_true")
    group.add_argument("--deck-only", action="store_true")
    args = parser.parse_args()

    for program in ("pdflatex", "bibtex"):
        if shutil.which(program) is None:
            raise SystemExit(f"required program not found: {program}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

    products: list[Path] = []
    if not args.deck_only:
        products.append(build_paper())
    if not args.paper_only:
        products.append(build_deck())
    for product in products:
        target = PUBLICATION_DIR / product.name
        shutil.copy2(product, target)
        print(f"published {target.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
