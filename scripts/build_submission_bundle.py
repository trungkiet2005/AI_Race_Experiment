"""Assemble the AAMAS submission: the paper PDF and the supplementary zip.

AAMAS takes supplementary material as a single zip of at most 25 MB, and the
review is double-blind, so nothing in the bundle may name an author, an
institution, a grant, or the public repository this work lives in.

The bundle is deliberately small. It carries the supplementary document and
nothing else: the code and the run artefacts live in a repository that is public
under the authors' own account, so shipping or linking them would identify the
authors to a reviewer.

The anonymity pattern list used to live in this file. It now lives in
``scripts/anonymity_scan.py`` and runs at every build of the PDFs as well as
here, because a gate that only runs at the last step of the process is a gate
most artefacts never meet: the leak that prompted the move went out as a PDF
that nobody had ever bundled.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):  # run as a script, so the sibling module is a plain import
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import anonymity_scan

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
BUNDLE = ROOT / "results" / "artifacts" / "submission"
LIMIT_BYTES = 25 * 1024 * 1024


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

    # Two questions, and they are not the same question. "Is this PDF clean" is
    # answered by scanning it here. "Was this PDF ever scanned when it was made"
    # is answered by its receipt, and only by its receipt: a PDF compiled by a
    # hand-run pdflatex, or copied in from another tree, reads exactly like one
    # the gate has passed. A receipt keyed to the wrong digest is not a receipt.
    hits: list[str] = []
    for pdf in (paper, supp):
        receipt = anonymity_scan.read_receipt(pdf)
        if receipt is None:
            raise SystemExit(
                f"{pdf.name} has no anonymity receipt for these exact bytes, so it "
                "was never scanned at the point it was made. Rebuild it with "
                "scripts/build_publication.py, which scans and leaves the receipt."
            )
        if receipt.get("overridden") and not args.allow_identifying:
            raise SystemExit(
                f"{pdf.name} was built with --allow-identifying, so its anonymity "
                "hits were waved through at build time. Fix the source and rebuild."
            )
        report = anonymity_scan.scan_pdf(pdf)
        hits.extend(report.hits)
        if not report.verified:
            hits.extend(f"{pdf.name}: {note}" for note in report.notes)

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
    # The copies are renamed, so they no longer carry the receipts written next
    # to their originals. Re-scan them under their bundle names: the point of a
    # receipt is that it names the bytes in front of you.
    for copy in (BUNDLE / "paper.pdf", BUNDLE / "supplementary.pdf"):
        copy_report = anonymity_scan.scan_pdf(copy)
        anonymity_scan.write_receipt(
            copy_report, overridden=args.allow_identifying and not copy_report.clean
        )
    zip_path = BUNDLE / "supplementary.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(supp, "supplementary.pdf")

    # The zip is the file the portal actually takes, so it is scanned after it
    # is written rather than trusted because its source was clean. Checking the
    # archive rather than its inputs also checks the member names: an archive
    # laid out under a home directory names a person before anyone opens it.
    # A dirty archive is deleted, not left on disk beside a "refusing" message,
    # because the next person to look in this folder would find a file that
    # looks finished.
    zip_report = anonymity_scan.scan_zip(zip_path)
    if not zip_report.clean and not args.allow_identifying:
        print("\n".join(zip_report.lines()))
        zip_path.unlink(missing_ok=True)
        raise SystemExit("the supplementary archive is not clean; removed it")
    anonymity_scan.write_receipt(zip_report, overridden=args.allow_identifying and not zip_report.clean)
    print("\n".join(zip_report.lines()))

    (BUNDLE / "README.txt").write_text(
        "AAMAS 2026 submission files\n"
        "===========================\n\n"
        "Upload to the portal:\n"
        "  paper.pdf          the manuscript, anonymous\n"
        "  supplementary.zip  supplementary material, single zip as the portal requires\n\n"
        "Not uploaded, here for reading only:\n"
        "  supplementary.pdf  the same document the zip contains\n"
        "  *.anonymity.json   the double-blind scan verdict for each file beside it,\n"
        "                     keyed to that file's SHA-256. Upload neither of these.\n\n"
        "Both PDFs build from paper/main.tex and paper/supplementary.tex through\n"
        "scripts/build_publication.py, which runs the same anonymity scan at the\n"
        "moment each PDF is made. Rebuild them before rebuilding this bundle.\n\n"
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
