"""Assemble the AAMAS submission: the paper PDF and the supplementary zip.

AAMAS takes supplementary material as a single zip of at most 25 MB, and the
review is double-blind, so nothing in the bundle may name an author, an
institution, a grant, or the public repository this work lives in.

The bundle is deliberately small. It carries the supplementary document and
nothing else: the code and the run artefacts live in a repository that is public
under the authors' own account, so shipping or linking them would identify the
authors to a reviewer.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
BUNDLE = ROOT / "results" / "artifacts" / "submission"
LIMIT_BYTES = 25 * 1024 * 1024

# Anything matching these must not appear in the bundle's text layer.
IDENTIFYING = [
    re.compile(r"hcmus\.edu\.vn", re.I),
    re.compile(r"hcmut\.edu\.vn", re.I),
    re.compile(r"\bVNU-?HCM\b", re.I),
    re.compile(r"\bEPSRC\b", re.I),
    re.compile(r"EP/Y00857X", re.I),
    re.compile(r"github\.com/[A-Za-z0-9_.-]+/AI_Race", re.I),
    re.compile(r"\bAI_Race_Experiment\b"),
]


def pdf_text(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        return ""
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout


def scan(path: Path) -> list[str]:
    text = pdf_text(path)
    if not text:
        return ["pdftotext unavailable, could not scan " + path.name]
    hits = []
    for pat in IDENTIFYING:
        for m in pat.finditer(text):
            line = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            hits.append(f"{path.name}: {pat.pattern} -> ...{line.strip()}...")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-identifying",
        action="store_true",
        help="build even if the anonymity scan finds something (never for a blind submission)",
    )
    args = parser.parse_args()

    paper = PAPER_DIR / "ai_race_paper.pdf"
    supp = PAPER_DIR / "ai_race_supplementary.pdf"
    for f in (paper, supp):
        if not f.is_file():
            raise SystemExit(f"missing {f}; run scripts/build_publication.py first")

    # This script copies a PDF; it does not compile one.  So a bundle can be
    # built from a PDF older than the manuscript that is supposed to be inside
    # it, and nothing downstream would notice: the file is valid, the anonymity
    # scan passes, and the size looks right.  That happened once, and the zip
    # that went out was missing a whole appendix while every gate reported clean.
    for pdf, source in ((paper, PAPER_DIR / "main.tex"),
                        (supp, PAPER_DIR / "supplementary.tex")):
        if source.is_file() and pdf.stat().st_mtime < source.stat().st_mtime:
            raise SystemExit(
                f"{pdf.name} is older than {source.name}, so the bundle would "
                "ship a manuscript that no longer matches its source. Run "
                "scripts/build_publication.py, then build the bundle again."
            )

    hits = scan(paper) + scan(supp)
    if hits and not args.allow_identifying:
        print("anonymity scan found identifying text:")
        for h in hits:
            print("   ", h)
        raise SystemExit("refusing to build the bundle")
    print("anonymity scan: clean" if not hits else "anonymity scan: OVERRIDDEN")

    BUNDLE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paper, BUNDLE / "paper.pdf")
    # A readable copy beside the archive: the zip is what the portal takes, but
    # nobody wants to unzip a file to check what they are about to upload.
    shutil.copy2(supp, BUNDLE / "supplementary.pdf")
    zip_path = BUNDLE / "supplementary.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(supp, "supplementary.pdf")

    (BUNDLE / "README.txt").write_text(
        "AAMAS 2026 submission files\n"
        "===========================\n\n"
        "Upload to the portal:\n"
        "  paper.pdf          the manuscript, anonymous\n"
        "  supplementary.zip  supplementary material, single zip as the portal requires\n\n"
        "Not uploaded, here for reading only:\n"
        "  supplementary.pdf  the same document the zip contains\n\n"
        "Both PDFs build from paper/main.tex and paper/supplementary.tex through\n"
        "scripts/build_publication.py. Rebuild them before rebuilding this bundle.\n\n"
        "Before the camera-ready, switch main.tex back to the non-anonymous\n"
        "\\documentclass[sigconf]{aamas} and restore the acknowledgements block\n"
        "that sits commented out near the end of the same file.\n",
        encoding="utf-8",
    )

    size = zip_path.stat().st_size
    print(f"paper           : {(BUNDLE / 'paper.pdf').stat().st_size / 1048576:.2f} MB")
    print(f"supplementary   : {size / 1048576:.2f} MB  (limit 25 MB)")
    if size > LIMIT_BYTES:
        raise SystemExit("supplementary zip exceeds the 25 MB limit")
    print(f"bundle written to {BUNDLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
