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
copy of the leaked document around. The account handles, the repository owner
and the home directory are read at run time from the same sources the scanner
reads (see ``scripts/anonymity_scan.py``), so this file names nobody either.

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

CLEAN_ROW = "Always Safe  account A  account B  account C  account D  account E"


def _pdf_string(value: str) -> bytes:
    """A PDF text string in hyperref's UTF-16BE octal form."""
    raw = b"\xfe\xff" + value.encode("utf-16-be")
    return b"(" + b"".join(b"\\%03o" % byte for byte in raw) + b")"


def minimal_pdf(path: Path, *, visible: str | list[str], link: str | None = None,
                info: dict[str, bytes] | None = None,
                form_info: dict[str, bytes] | None = None) -> Path:
    """A one-page PDF with ``visible`` in its text layer.

    ``link`` goes only into a link annotation, which is what a reader clicks and
    what ``pdftotext`` never prints. ``info`` becomes the document information
    dictionary and ``form_info`` the ``PTEX.InfoDict`` of an embedded form
    XObject, the two places an included figure's metadata ends up.
    """
    rows = [visible] if isinstance(visible, str) else visible
    body = b"BT /F1 11 Tf 14 TL 60 740 Td " + b" ".join(
        f"({row}) Tj T*".encode("latin-1") for row in rows) + b" ET\n"
    if form_info is not None:
        body += b"q /Fx Do Q\n"
    annots = ""
    if link is not None:
        annots = ("/Annots[<</Type/Annot/Subtype/Link/Rect[60 690 300 714]"
                  f"/Border[0 0 0]/A<</Type/Action/S/URI/URI({link})>>>>]")
    xobject = "/XObject<</Fx 6 0 R>>" if form_info is not None else ""
    form_body = b"0 0 m 1 1 l S\n"
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        ("<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 4 0 R>>"
         f"{xobject}>>{annots}/Contents 5 0 R>>").encode("latin-1"),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        b"<</Length " + str(len(body)).encode() + b">>\nstream\n" + body + b"endstream",
    ]
    if form_info is not None:
        entries = b"".join(k.encode() + v for k, v in form_info.items())
        objects.append(b"<</Type/XObject/Subtype/Form/BBox[0 0 10 10]/PTEX.InfoDict 7 0 R/Length "
                       + str(len(form_body)).encode() + b">>\nstream\n" + form_body + b"endstream")
        objects.append(b"<<" + entries + b">>")
    info_ref = b""
    if info is not None:
        objects.append(b"<<" + b"".join(k.encode() + v for k, v in info.items()) + b">>")
        info_ref = f"/Info {len(objects)} 0 R".encode()
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
    out += (f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R".encode() + info_ref
            + f">>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(bytes(out))
    return path


def expect(condition: bool, description: str, failures: list[str]) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {description}")
    if not condition:
        failures.append(description)


def _hits_for(report: anonymity_scan.ScanReport, origin: str, label: str) -> int:
    return sum(1 for hit in report.hits if origin in hit and label in hit)


def main() -> int:
    failures: list[str] = []
    if shutil.which("pdftotext") is None:
        print("pdftotext is not on PATH; the text-layer half of the gate cannot be checked")
        return 1

    ids = anonymity_scan.load_identities()
    print("the identity lists the scanner reads at run time")
    expect(not ids.problems, "every identity source was found: " + "; ".join(ids.sources), failures)
    expect(len(ids.accounts) >= 5, f"{len(ids.accounts)} compute accounts loaded", failures)
    expect(any(v == "author name" for v in ids.people.values()),
           "author names were parsed from the manuscript's author block", failures)
    owner = next((t for t, v in ids.tokens.items() if v == "repository owner"), None)
    repo = next((t for t, v in ids.tokens.items() if v == "repository name"), None)
    if not ids.accounts:
        print("\nno account list, so the leak cannot be replayed")
        return 1

    # The scanner's own source may show a generic shape as an example (a home
    # path, a grant number); it must not contain any loaded identity.
    generic = {label for label, _ in anonymity_scan.GENERIC}
    for own in ("anonymity_scan.py", "check_anonymity_gate.py"):
        source = (Path(__file__).resolve().parent / own).read_text(encoding="utf-8")
        leaks = [h for h in anonymity_scan.scan_text(source, own)
                 if h.split(": ", 1)[1].split(" -> ")[0] not in generic]
        expect(not leaks, f"{own} spells out no account, author, institution or owner", failures)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        print("the leak as it shipped: the accounts printed in a supplementary table")
        handles = ids.accounts
        leaked = minimal_pdf(tmp / "leaked.pdf", visible=["Always Safe"] + handles)
        report = anonymity_scan.scan_pdf(leaked)
        expect(report.verified, "the PDF was actually read, so the verdict means something", failures)
        expect(not report.clean, "the scan refuses the leaked table", failures)
        caught = _hits_for(report, "[text layer]", "compute account")
        expect(caught >= len(handles), f"all {len(handles)} accounts are caught in the text layer ({caught})",
               failures)
        expect(not any(h in hit for h in handles for hit in report.hits),
               "the hits are redacted, so a receipt never repeats a handle", failures)

        print("the same table with the accounts replaced")
        clean = minimal_pdf(tmp / "clean.pdf", visible=CLEAN_ROW)
        clean_report = anonymity_scan.scan_pdf(clean)
        expect(clean_report.clean, "the scan passes the corrected table", failures)

        print("a leak the text layer cannot see: the repository only in a link target")
        if owner and repo:
            linked = minimal_pdf(tmp / "linked.pdf", visible="our code is available online",
                                 link=f"https://github.com/{owner}/{repo}")
            text = anonymity_scan.pdf_text(linked) or ""
            expect("github" not in text.lower(), "pdftotext does indeed not print the link target", failures)
            expect(not anonymity_scan.scan_pdf(linked).clean,
                   "the object-surface scan catches it anyway", failures)
        else:
            expect(False, "git metadata names the repository owner and name", failures)

        print("a home-directory path, the category the repository sweep found missing")
        home = minimal_pdf(tmp / "home.pdf", visible=f"run from {Path.home()}")
        expect(not anonymity_scan.scan_pdf(home).clean, "the scan refuses a home-directory path", failures)

        print("metadata a reader never sees on the page")
        keywords = minimal_pdf(tmp / "keywords.pdf", visible="a figure",
                               form_info={"/Creator": b"(design tool)", "/Keywords": b"(XYZ123,ABC987)"})
        report = anonymity_scan.scan_pdf(keywords)
        expect(any("/Keywords" in hit and "XObject" in hit for hit in report.hits),
               "non-empty /Keywords in an embedded figure's information dictionary is refused", failures)
        author = minimal_pdf(tmp / "author.pdf", visible="nothing here",
                             info={"/Author": _pdf_string(handles[0])})
        report = anonymity_scan.scan_pdf(author)
        expect(any("document /Author" in hit and "compute account" in hit for hit in report.hits),
               "an account in a UTF-16 /Author is decoded and caught", failures)
        hex_author = minimal_pdf(
            tmp / "hex.pdf", visible="nothing here",
            info={"/Subject": b"<FEFF" + handles[1].encode("utf-16-be").hex().upper().encode() + b">"})
        report = anonymity_scan.scan_pdf(hex_author)
        expect(any("/Subject" in hit and "compute account" in hit for hit in report.hits),
               "an account in a UTF-16 hex /Subject is decoded and caught", failures)
        anonymous = minimal_pdf(tmp / "anon.pdf", visible="nothing here",
                                info={"/Author": _pdf_string("Anonymous Author(s)"),
                                      "/Title": b"(A title)"})
        expect(anonymity_scan.scan_pdf(anonymous).clean,
               "an anonymous /Author and a plain /Title pass", failures)

        print("the archive, not only the PDF beside it")
        dirty_zip = tmp / "supplementary.zip"
        with zipfile.ZipFile(dirty_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(leaked, "supplementary.pdf")
        zip_report = anonymity_scan.scan_zip(dirty_zip)
        expect(zip_report.verified, "the archive was opened and read", failures)
        expect(not zip_report.clean, "a clean-looking zip holding a dirty PDF is refused", failures)
        named_zip = tmp / "named.zip"
        with zipfile.ZipFile(named_zip, "w") as z:
            z.writestr(f"{Path.home().as_posix()}/supplementary.txt", "nothing identifying inside")
        expect(not anonymity_scan.scan_zip(named_zip).clean,
               "a member path that names a person is refused even when its contents are clean", failures)

        print("a code snapshot, scanned as a directory")
        tree = tmp / "snapshot"
        (tree / "results").mkdir(parents=True)
        (tree / "results" / "receipt.json").write_text('{"executing_identity": "account_A"}\n', encoding="utf-8")
        expect(anonymity_scan.scan_tree(tree).clean, "a pseudonymised receipt passes", failures)
        (tree / "results" / f"attempt_{handles[2]}_20260910.json").write_text("{}\n", encoding="utf-8")
        expect(not anonymity_scan.scan_tree(tree).clean,
               "an account glued into a file name by underscores is refused", failures)

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
