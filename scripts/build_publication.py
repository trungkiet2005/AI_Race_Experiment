"""Compile the paper and deck into the project publication artifact folders.

LaTeX auxiliary files are isolated under ``results/_build/latex/current`` so a
repository build never recreates the retired top-level ``output/`` tree. The
final paper and supplementary PDFs are written to ``paper/`` for easy access;
the publication artifact tree keeps a synchronized copy for release tooling.

The two documents disagree about what their asset paths are relative to:
``paper/main.tex`` loads ``aamas.cls`` and ``figures/...`` relative to ``paper/``,
while ``slides/ai_race_research_deck.tex`` reaches for ``paper/figures/...`` and
``results/...`` relative to the repository root. Each is therefore compiled from
its own directory with TEXINPUTS extended to the repository root, so both
conventions resolve without editing either document.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "results" / "_build" / "latex" / "current"
PAPER_DIR = ROOT / "paper"
PUBLICATION_DIR = ROOT / "results" / "artifacts" / "publication"


def run(command: list[str], *, cwd: Path = ROOT, bib_dir: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    # A trailing separator keeps kpathsea's own defaults on the end of the list.
    env = dict(os.environ)
    env["TEXINPUTS"] = f"{ROOT}{os.pathsep}{env.get('TEXINPUTS', '')}{os.pathsep}"
    if bib_dir is not None:
        # bibtex resolves \bibdata against BIBINPUTS, not TEXINPUTS, and it does
        # not fail when it cannot find the database: it reports "didn't find a
        # database entry" once per citation and returns success, which reaches
        # the PDF as [?] rather than as a build error.
        env["BIBINPUTS"] = f"{bib_dir}{os.pathsep}{env.get('BIBINPUTS', '')}{os.pathsep}"
    subprocess.run(command, cwd=cwd, check=True, env=env)


def latex(source: str, *, jobname: str) -> None:
    document = ROOT / source
    run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={BUILD_DIR.as_posix()}",
            f"-jobname={jobname}",
            document.name,
        ],
        cwd=document.parent,
    )


def build_paper() -> Path:
    latex("paper/main.tex", jobname="ai_race_paper")
    run(
        ["bibtex", (BUILD_DIR / "ai_race_paper").as_posix()],
        bib_dir=ROOT / "paper",
    )
    latex("paper/main.tex", jobname="ai_race_paper")
    latex("paper/main.tex", jobname="ai_race_paper")
    return BUILD_DIR / "ai_race_paper.pdf"


def build_supplement() -> Path:
    latex("paper/supplementary.tex", jobname="ai_race_supplementary")
    run(
        ["bibtex", (BUILD_DIR / "ai_race_supplementary").as_posix()],
        bib_dir=ROOT / "paper",
    )
    latex("paper/supplementary.tex", jobname="ai_race_supplementary")
    latex("paper/supplementary.tex", jobname="ai_race_supplementary")
    return BUILD_DIR / "ai_race_supplementary.pdf"


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
    if BUILD_DIR.exists():
        if BUILD_DIR.is_symlink() or not BUILD_DIR.is_dir():
            raise SystemExit(f"refusing to remove non-directory build path: {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

    products: list[Path] = []
    if not args.deck_only:
        products.append(build_paper())
        products.append(build_supplement())
    if not args.paper_only:
        products.append(build_deck())
    for product in products:
        if product.name in {"ai_race_paper.pdf", "ai_race_supplementary.pdf"}:
            target = PAPER_DIR / product.name
            shutil.copy2(product, target)
            print(f"published {target.relative_to(ROOT)}", flush=True)
            mirror = PUBLICATION_DIR / product.name
            shutil.copy2(target, mirror)
            print(f"synced {mirror.relative_to(ROOT)}", flush=True)
        else:
            target = PUBLICATION_DIR / product.name
            shutil.copy2(product, target)
            print(f"published {target.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
