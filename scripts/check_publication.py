#!/usr/bin/env python3
"""Run the machine-checkable publication gates for the paper PDFs.

The compiler is allowed to emit non-fatal warnings during drafting. This gate
keeps the defects that can invalidate a submission explicit: missing or
malformed PDFs, fatal LaTeX errors, unresolved references or citations,
overfull boxes, Type 3 fonts, and the unassigned anonymous submission ID.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
BUILD_DIR = ROOT / "results" / "_build" / "latex" / "current"
PDF_NAMES = ("ai_race_paper.pdf", "ai_race_supplementary.pdf")


def run_tool(command: list[str]) -> tuple[int, str]:
    if shutil.which(command[0]) is None:
        return 127, ""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def check_pdf(pdf: Path, *, allow_placeholder_id: bool) -> list[str]:
    errors: list[str] = []
    if not pdf.is_file():
        return [f"missing PDF: {pdf.relative_to(ROOT)}"]
    if pdf.read_bytes()[:5] != b"%PDF-":
        errors.append(f"invalid PDF header: {pdf.relative_to(ROOT)}")

    stem = pdf.stem
    log = BUILD_DIR / f"{stem}.log"
    if not log.is_file():
        errors.append(f"missing LaTeX log: {log.relative_to(ROOT)}")
    else:
        log_text = log.read_text(encoding="utf-8", errors="replace")
        checks = (
            (r"^!", "fatal LaTeX error"),
            (r"Citation .* undefined", "undefined citation"),
            (r"Reference .* undefined", "undefined reference"),
            (r"There were undefined references", "undefined references"),
            (r"Overfull \\hbox", "overfull hbox"),
        )
        for pattern, label in checks:
            if re.search(pattern, log_text, flags=re.MULTILINE | re.IGNORECASE):
                errors.append(f"{label} in {log.relative_to(ROOT)}")

    code, fonts = run_tool(["pdffonts", str(pdf)])
    if code == 127:
        errors.append("pdffonts is unavailable")
    elif code != 0:
        errors.append(f"pdffonts failed for {pdf.relative_to(ROOT)}")
    elif re.search(r"^.*Type 3", fonts, flags=re.MULTILINE):
        errors.append(f"Type 3 font in {pdf.relative_to(ROOT)}")

    code, text = run_tool(["pdftotext", "-layout", str(pdf), "-"])
    if code == 127:
        errors.append("pdftotext is unavailable")
    elif code != 0:
        errors.append(f"pdftotext failed for {pdf.relative_to(ROOT)}")
    else:
        if "??" in text or "[?]" in text:
            errors.append(f"unresolved marker in {pdf.relative_to(ROOT)}")
        if not allow_placeholder_id and re.search(r"Submission Id:\s*TBD", text):
            errors.append(f"submission ID is still TBD in {pdf.relative_to(ROOT)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-placeholder-id",
        action="store_true",
        help="allow the local draft's TBD anonymous submission ID",
    )
    args = parser.parse_args()

    errors: list[str] = []
    for name in PDF_NAMES:
        errors.extend(
            check_pdf(PAPER_DIR / name, allow_placeholder_id=args.allow_placeholder_id)
        )

    if errors:
        print("publication QA failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("publication QA passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
