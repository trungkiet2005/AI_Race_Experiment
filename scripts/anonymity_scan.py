"""The anonymity gate: one pattern list, one scanner, every build.

The review is double-blind, so nothing in a manuscript PDF or in the
supplementary archive may name an author, an institution, a grant, the public
repository this work lives in, or the compute accounts the runs executed on.

This module exists because that rule was enforced in the wrong place. The scan
used to live inside ``scripts/build_submission_bundle.py``, which is the last
step of the process and the one that runs least often. ``build_publication.py``
makes the PDFs and never scanned them at all, so a PDF could be compiled, read,
mailed to an external reader and discussed without ever meeting the gate that
exists to protect it. That is exactly what happened: a supplementary table
printed the five collaborator accounts the scripted-opponent grid ran on, the
PDF went out, and the bundle scan that would have caught it was never reached
because nobody was building a bundle.

So the pattern list lives here, on its own, and both builds import it. One list,
because two copies drift and the copy that drifts is the one that is checked.

Two surfaces are read, not one:

* the text layer, through ``pdftotext``, which is what a reviewer reads;
* the PDF objects, inflated here, which is what a reviewer's PDF viewer reads.

The second surface is not redundant. ``pdftotext`` prints the anchor text of a
hyperlink and never the target, so ``\\href{https://github.com/<account>/...}``
{our code} is invisible to a text-layer scan and perfectly visible to anyone who
clicks it. Document metadata is invisible to a text-layer scan for the same
reason.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

# Bumped whenever the pattern list or the scanned surface changes, so a receipt
# written by an older scanner is not mistaken for a verdict from this one.
SCANNER_VERSION = 3

RECEIPT_SUFFIX = ".anonymity.json"

# Anything matching these must not appear in an artefact a reviewer receives.
IDENTIFYING: list[re.Pattern[str]] = [
    re.compile(r"hcmus\.edu\.vn", re.I),
    re.compile(r"hcmut\.edu\.vn", re.I),
    re.compile(r"\bVNU-?HCM\b", re.I),
    re.compile(r"\bEPSRC\b", re.I),
    re.compile(r"EP/Y00857X", re.I),
    re.compile(r"github\.com/[A-Za-z0-9_.-]+/AI_Race", re.I),
    re.compile(r"\bAI_Race_Experiment\b"),
    # The compute accounts. These are the identifiers most likely to survive a
    # rewrite, because they are written down as provenance rather than as
    # authorship, and a reader who searches one finds a person. The scan missed
    # exactly this once: a supplement subsection named the account the
    # admission campaign ran on, and the bundle was reported clean.
    re.compile(r"\bdaosyduyminh\b", re.I),
    re.compile(r"\bfoundnotkiet\b", re.I),
    re.compile(r"\bhunhtrungkit\b", re.I),
    re.compile(r"\btnkiet\b", re.I),
    re.compile(r"\bkit567\b", re.I),
    re.compile(r"\btrungkiet\b", re.I),
    # Author surnames are deliberately NOT listed. Two of them wrote the source
    # study this paper builds on, and citing that work in the third person is
    # both standard and required; a scan that flagged it would be telling the
    # author to drop a citation in order to look anonymous.
    #
    # The rest of this list was added after a repository-wide sweep showed which
    # identifying categories the original six accounts did not cover.
    #
    # Hosting accounts. The five compute accounts above are the ones that reach
    # a run receipt, but the accounts that own the repository, the Kaggle
    # datasets and the kernels are just as identifying and were not matched:
    # "\btrungkiet\b" does not match "trungkiet2005", because the boundary it
    # asks for falls between two word characters and therefore does not exist.
    re.compile(r"\btrungkiet2005\b", re.I),
    re.compile(r"\btechnoob05\b", re.I),
    re.compile(r"\blphonghao193\b", re.I),
    re.compile(r"\bnguyenlamphuquy\b", re.I),
    # Home-directory paths. This is the category the sweep found missing. A
    # LaTeX console dump, a traceback quoted into an appendix or a "how we ran
    # it" listing carries the machine account name in every line, and the
    # account name here is a person's name. The drive-letter form and the POSIX
    # form are both in the tree today, in build logs and in analysis reports.
    re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\\/\s\"')\]]+", re.I),
    re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+", re.I),
    re.compile(r"\bPhD_Farming\b|\bPhD_LetGoo\b", re.I),
    # Personal mail. Deliberately restricted to consumer providers and to the
    # institutional domains already listed above, rather than a generic address
    # regex: the reference list of a paper in this area quotes other people's
    # institutional addresses, and a gate that fires on a correct citation is a
    # gate somebody switches off.
    re.compile(r"[A-Za-z0-9._%+-]+@(?:gmail|googlemail|outlook|hotmail|yahoo|icloud|proton(?:mail)?)\.[A-Za-z.]{2,}", re.I),
]


class ScanReport:
    """What one artefact's scan concluded, and whether it concluded anything."""

    def __init__(self, target: Path):
        self.target = target
        self.hits: list[str] = []
        self.notes: list[str] = []
        # ``verified`` answers "was this artefact actually read", which is a
        # different question from "was it clean". A scanner that could not run
        # must never be recorded as a pass; silence is not a clean verdict.
        self.verified = False

    @property
    def clean(self) -> bool:
        return self.verified and not self.hits

    def lines(self) -> list[str]:
        out = [f"anonymity scan: {self.target.name}"]
        for note in self.notes:
            out.append(f"    note: {note}")
        if not self.verified:
            out.append("    NOT VERIFIED: the artefact could not be read, so it has no verdict")
        elif self.hits:
            out.append(f"    {len(self.hits)} identifying hit(s):")
            out.extend(f"      {h}" for h in self.hits)
        else:
            out.append("    clean")
        return out


def _context(text: str, start: int, end: int) -> str:
    window = text[max(0, start - 60):end + 60].replace("\n", " ").replace("\r", " ")
    return " ".join(window.split())


def scan_text(text: str, origin: str) -> list[str]:
    """Every pattern hit in one blob of text, each with its surrounding words."""
    hits: list[str] = []
    for pat in IDENTIFYING:
        for m in pat.finditer(text):
            hits.append(f"{origin}: {pat.pattern} -> ...{_context(text, m.start(), m.end())}...")
    return hits


def pdf_text(path: Path) -> str | None:
    """The text layer, or None when ``pdftotext`` is not available to read it."""
    if shutil.which("pdftotext") is None:
        return None
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
        return None
    return out.stdout


# An embedded font carries its own licence text, and the licence of the font
# this paper sets its body in contains the placeholder address
# "firstname.lastname@gmail.com". A font program is not authored content and
# never carries an author's identity, so it is excluded from the object scan.
# Without this the gate fires three times on every build of a clean paper, and a
# gate that cries wolf on correct output is a gate somebody turns off.
_FONT_MARKERS = (b"eexec", b"%!PS-Adobe", b"/CharStrings", b"/FontMatrix", b"/FontName")


def _is_font_program(payload: bytes) -> bool:
    head = payload[:4096]
    return any(marker in head for marker in _FONT_MARKERS)


def pdf_objects(path: Path) -> str:
    """The PDF's object dictionaries plus every stream this can inflate.

    Link targets, embedded file names and document metadata live in the object
    graph and never reach the text layer. Streams that will not inflate are kept
    as they are rather than raised on: an image is not expected to decompress,
    and failing a build over one would teach the reader to pass --allow.
    """
    data = path.read_bytes()
    parts: list[bytes] = []
    cursor = 0
    for m in re.finditer(rb"stream\r?\n", data):
        # A match inside a payload already consumed is a coincidence in binary
        # data, not a stream header; taking it would mis-slice everything after.
        if m.start() < cursor:
            continue
        end = data.find(b"endstream", m.end())
        if end < 0:
            continue
        parts.append(data[cursor:m.start()])
        cursor = end
        payload = data[m.end():end]
        try:
            payload = zlib.decompressobj().decompress(payload)
        except zlib.error:
            pass
        if not _is_font_program(payload):
            parts.append(payload)
    parts.append(data[cursor:])
    return b"\n".join(parts).decode("latin-1", "replace")


def scan_pdf(path: Path) -> ScanReport:
    report = ScanReport(path)
    text = pdf_text(path)
    if text is None:
        report.notes.append(
            "pdftotext is not on PATH; install poppler-utils so the text layer "
            "a reviewer reads can be checked"
        )
    else:
        report.verified = True
        report.hits.extend(scan_text(text, f"{path.name} [text layer]"))
    try:
        objects = pdf_objects(path)
    except OSError as exc:
        report.notes.append(f"could not read the PDF objects: {exc}")
    else:
        report.hits.extend(scan_text(objects, f"{path.name} [pdf objects]"))
        report.notes.append("pdf objects inflated and scanned")
    return report


def scan_zip(path: Path) -> ScanReport:
    """Scan an archive member by member.

    A clean PDF sitting beside a dirty archive is still a leak, and the archive
    is the file the portal actually takes.
    """
    report = ScanReport(path)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        report.notes.append(f"could not open the archive: {exc}")
        return report
    with archive, tempfile.TemporaryDirectory() as tmp:
        report.verified = True
        for name in archive.namelist():
            # A path can identify as readily as its contents: an archive whose
            # members sit under C:/Users/<person>/ needs no reading at all.
            report.hits.extend(scan_text(name, f"{path.name} [member name]"))
            if name.endswith("/"):
                continue
            extracted = Path(archive.extract(name, tmp))
            if extracted.suffix.lower() == ".pdf":
                inner = scan_pdf(extracted)
                report.hits.extend(f"{path.name} > {h}" for h in inner.hits)
                if not inner.verified:
                    report.verified = False
                    report.notes.extend(inner.notes)
                continue
            try:
                body = extracted.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                report.notes.append(f"{name}: not text, only the member name was scanned")
                continue
            report.hits.extend(scan_text(body, f"{path.name} > {name}"))
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_path(artefact: Path) -> Path:
    return artefact.with_suffix(artefact.suffix + RECEIPT_SUFFIX)


def write_receipt(report: ScanReport, *, overridden: bool = False) -> Path:
    """Bind the verdict to the bytes it was reached on.

    A verdict that lives only in build output is a verdict nobody has when it
    matters. The receipt travels with the artefact and carries the artefact's
    digest, so a PDF produced by some other route -- a hand-run pdflatex, a copy
    from an old tree -- has no receipt that matches it, and the bundle build can
    tell "scanned and clean" apart from "never scanned".
    """
    path = receipt_path(report.target)
    path.write_text(
        json.dumps(
            {
                "artefact": report.target.name,
                "sha256": sha256(report.target),
                "scanner_version": SCANNER_VERSION,
                "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "verified": report.verified,
                "clean": report.clean,
                "overridden": overridden,
                "notes": report.notes,
                "hits": report.hits,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def read_receipt(artefact: Path) -> dict | None:
    """The receipt for this exact artefact, or None if there is not one.

    "Not one" includes a receipt for different bytes and a receipt from an older
    scanner, both of which describe a PDF that is not the PDF in hand.
    """
    path = receipt_path(artefact)
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if receipt.get("scanner_version") != SCANNER_VERSION:
        return None
    if receipt.get("sha256") != sha256(artefact):
        return None
    return receipt


def scan_artefact(path: Path) -> ScanReport:
    return scan_zip(path) if path.suffix.lower() == ".zip" else scan_pdf(path)


def main(argv: list[str] | None = None) -> int:
    """Scan the artefacts named on the command line. Non-zero if any is not clean."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Scan PDFs and zips for identifying text.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--receipt", action="store_true", help="write a scan receipt beside each artefact")
    args = parser.parse_args(argv)

    worst = 0
    for path in args.paths:
        report = scan_artefact(path)
        print("\n".join(report.lines()))
        if args.receipt:
            write_receipt(report)
        if not report.clean:
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
