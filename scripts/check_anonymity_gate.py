#!/usr/bin/env python3
"""Prove the anonymity gate still catches the leak it was built for.

A gate is only worth the confidence placed in it if it has been watched failing
on the thing it missed. The leak this checks for is the real one: a supplementary
table that printed the five collaborator accounts the scripted-opponent grid ran
on, in a PDF that was compiled, read and mailed to an external reader without
ever meeting a scan, because the only scan in the process lived in the bundle
build and nobody was building a bundle.

The fixtures here are built from scratch rather than kept as a copy of that PDF,
so the check is self-contained, costs no repository weight, and keeps no live
copy of the leaked document around.

Run it after touching scripts/anonymity_scan.py, scripts/build_publication.py or
scripts/build_submission_bundle.py.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import anonymity_scan

# The exact row shape from the leaked supplementary table.
LEAKED_ROW = "Always Safe  kit567  hunhtrungkit  tnkiet  trungkiet  foundnotkiet"
CLEAN_ROW = "Always Safe  account A  account B  account C  account D  account E"


def minimal_pdf(path: Path, *, visible: str, link: str | None = None) -> Path:
    """A one-page PDF with ``visible`` in its text layer and ``link`` as a URI.

    ``link`` goes only into a link annotation, which is what a reader clicks and
    what ``pdftotext`` never prints. It is here so the check can show that the
    object-surface scan is not decoration.
    """
    body = f"BT /F1 11 Tf 60 700 Td ({visible}) Tj ET\n".encode("latin-1")
    annots = ""
    if link is not None:
        annots = ("/Annots[<</Type/Annot/Subtype/Link/Rect[60 690 300 714]"
                  f"/Border[0 0 0]/A<</Type/Action/S/URI/URI({link})>>>>]")
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        ("<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 4 0 R>>>>"
         f"{annots}/Contents 5 0 R>>").encode("latin-1"),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        b"<</Length " + str(len(body)).encode() + b">>\nstream\n" + body + b"endstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def expect(condition: bool, description: str, failures: list[str]) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {description}")
    if not condition:
        failures.append(description)


def main() -> int:
    failures: list[str] = []
    if shutil.which("pdftotext") is None:
        print("pdftotext is not on PATH; the text-layer half of the gate cannot be checked")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        print("the leak as it shipped: five accounts printed in a supplementary table")
        leaked = minimal_pdf(tmp / "leaked.pdf", visible=LEAKED_ROW)
        report = anonymity_scan.scan_pdf(leaked)
        expect(report.verified, "the PDF was actually read, so the verdict means something", failures)
        expect(not report.clean, "the scan refuses the leaked table", failures)
        for handle in ("kit567", "hunhtrungkit", "tnkiet", "trungkiet", "foundnotkiet"):
            expect(any(handle in hit for hit in report.hits), f"{handle} is named in the hits", failures)

        print("the same table with the accounts replaced")
        clean = minimal_pdf(tmp / "clean.pdf", visible=CLEAN_ROW)
        clean_report = anonymity_scan.scan_pdf(clean)
        expect(clean_report.clean, "the scan passes the corrected table", failures)

        print("a leak the text layer cannot see: an account only in a link target")
        linked = minimal_pdf(tmp / "linked.pdf", visible="our code is available online",
                             link="https://github.com/trungkiet2005/AI_Race_Experiment")
        text = anonymity_scan.pdf_text(linked) or ""
        expect("github" not in text.lower(), "pdftotext does indeed not print the link target", failures)
        link_report = anonymity_scan.scan_pdf(linked)
        expect(not link_report.clean, "the object-surface scan catches it anyway", failures)

        print("a home-directory path, the category the repository sweep found missing")
        home = minimal_pdf(tmp / "home.pdf", visible=r"run from C:\Users\DELL 3450\PhD_LetGoo")
        expect(not anonymity_scan.scan_pdf(home).clean, "the scan refuses a home-directory path", failures)

        print("the archive, not only the PDF beside it")
        dirty_zip = tmp / "supplementary.zip"
        with zipfile.ZipFile(dirty_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(leaked, "supplementary.pdf")
        zip_report = anonymity_scan.scan_zip(dirty_zip)
        expect(zip_report.verified, "the archive was opened and read", failures)
        expect(not zip_report.clean, "a clean-looking zip holding a dirty PDF is refused", failures)
        named_zip = tmp / "named.zip"
        with zipfile.ZipFile(named_zip, "w") as z:
            z.writestr("C:/Users/DELL 3450/supplementary.txt", "nothing identifying inside")
        expect(not anonymity_scan.scan_zip(named_zip).clean,
               "a member path that names a person is refused even when its contents are clean", failures)

        print("the receipt tells 'scanned and clean' apart from 'never scanned'")
        anonymity_scan.write_receipt(clean_report)
        expect(anonymity_scan.read_receipt(clean) is not None,
               "a receipt written for these bytes is accepted", failures)
        clean.write_bytes(clean.read_bytes() + b"% edited after the scan\n")
        expect(anonymity_scan.read_receipt(clean) is None,
               "a receipt stops matching as soon as the PDF changes", failures)
        expect(anonymity_scan.read_receipt(leaked) is None,
               "a PDF that was never scanned has no receipt at all", failures)

    print()
    if failures:
        print(f"anonymity gate check FAILED: {len(failures)} of its own assertions did not hold")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("anonymity gate check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
