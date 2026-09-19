"""Compile the paper and deck into the project publication artifact folders.

LaTeX auxiliary files are isolated under ``results/_build/latex/current`` so a
repository build never recreates the retired top-level ``output/`` tree. The
final paper and supplementary PDFs are written to ``paper/`` for easy access;
the publication artifact tree keeps a synchronized copy for release tooling.

The two documents disagree about what their asset paths are relative to:
``paper/main.tex`` loads ``aamas.cls`` and ``figures/...`` relative to ``paper/``,
while ``slides/ai_race_research_deck.tex`` reaches for ``figures/paper/...``
relative to the repository root. Each is compiled from its own directory with
TEXINPUTS extended to the repository root, so the paper and deck share one
canonical figure hub.

Every manuscript PDF this script publishes has its Info and XMP metadata, and
that of every embedded figure, stripped (``strip_metadata``), and is then put
through the double-blind anonymity scan before the script returns, and each one
gets a scan receipt
written beside it. See ``scripts/anonymity_scan.py`` for why the gate moved here
and ``check_anonymity`` below for why a hit does not abort the build.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script, so the sibling module is a plain import
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import anonymity_scan


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


def build_figures() -> None:
    """Regenerate the canonical manuscript figures before LaTeX compilation."""

    run([sys.executable, str(ROOT / "scripts" / "build_publication_figures.py")], cwd=ROOT)


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


MANUSCRIPTS = {"ai_race_paper.pdf", "ai_race_supplementary.pdf"}

# Everything in these is a statement about who made the file, on what machine
# and when, and none of it is part of the paper. The title stays: it is on the
# first page anyway, and a viewer shows it in the tab.
_INFO_KEYS = ("/Author", "/Keywords", "/Subject", "/Creator", "/Producer",
              "/CreationDate", "/ModDate", "/Trapped", "/PTEX.Fullbanner")
_PRIVATE_KEYS = ("/PTEX.InfoDict", "/Metadata", "/PieceInfo")


def strip_metadata(pdf: Path) -> int:
    """Remove Info and XMP metadata from a final PDF and its embedded figures.

    ``\\includegraphics`` copies each figure's own information dictionary into
    the output as ``/PTEX.InfoDict``. One protected artwork was exported from a
    design tool that writes its design and team identifiers into ``/Keywords``,
    and those travelled into the supplement. The artwork itself is protected and
    is never rewritten, so the metadata is removed here, from the compiled PDF,
    after compilation and before the anonymity scan reads it. The build still
    validates the protected sources exactly as before.

    Returns the number of embedded dictionaries removed.
    """
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import IndirectObject
    except ImportError as exc:
        raise SystemExit("pypdf is required to strip PDF metadata: pip install pypdf") from exc

    reader = PdfReader(str(pdf))
    writer = PdfWriter(clone_from=reader)
    writer.pdf_header = reader.pdf_header
    title = (reader.metadata or {}).get("/Title")
    writer.metadata = None
    if title:
        writer.add_metadata({"/Title": title})
    for key in ("/Metadata", "/PieceInfo"):
        if key in writer.root_object:
            del writer.root_object[key]

    seen: set[int] = set()
    removed = 0

    def walk(resources) -> None:
        nonlocal removed
        if resources is None:
            return
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            return
        for ref in xobjects.get_object().values():
            if isinstance(ref, IndirectObject):
                if ref.idnum in seen:
                    continue
                seen.add(ref.idnum)
            obj = ref.get_object()
            if obj.get("/Subtype") != "/Form":
                continue
            for key in _PRIVATE_KEYS:
                if key in obj:
                    del obj[key]
                    removed += 1
            walk(obj.get("/Resources"))

    for page in writer.pages:
        for key in ("/Metadata", "/PieceInfo"):
            if key in page:
                del page[key]
        walk(page.get("/Resources"))
    # Deleting a reference leaves the dictionary it pointed to in the object
    # table; without this the Canva identifiers would still be in the bytes.
    writer.compress_identical_objects(remove_duplicates=False, remove_unreferenced=True)
    tmp = pdf.with_suffix(".stripped.pdf")
    with tmp.open("wb") as handle:
        writer.write(handle)
    os.replace(tmp, pdf)

    leftover = PdfReader(str(pdf)).metadata or {}
    if any(key in leftover for key in _INFO_KEYS) or "/Metadata" in PdfReader(str(pdf)).trailer["/Root"]:
        raise SystemExit(f"metadata survived stripping in {pdf.name}")
    return removed


def check_anonymity(pdfs: list[Path], *, allow: bool) -> bool:
    """Scan every published manuscript PDF and leave a receipt beside each.

    Fail or warn was a real decision, so it is written down here.

    Refusing to write the PDF was rejected. The manuscript is compiled dozens of
    times a day while it is being written, and half of those builds legitimately
    contain a name: the camera-ready author block lives commented out in
    ``main.tex`` and gets uncommented to check a layout. A gate that deletes the
    drafter's output is a gate the drafter removes, and then there is no gate.

    A warning printed and forgotten was rejected for the opposite reason. That
    is precisely the failure this code exists to fix: the supplementary table
    naming the five collaborator accounts was compiled, read, and mailed to an
    external reader, and nothing in the process ever said a word about it.

    So the build produces the PDF and then fails. The artefact is there for the
    drafter, and the command that made it reports non-zero, which is what a
    script, a Makefile, a hook or a CI job reads. Between those two, the receipt
    written next to each PDF carries the verdict in the artefact's own
    neighbourhood and is keyed to the artefact's digest, so a PDF that was never
    scanned has no receipt that matches it and ``build_submission_bundle.py``
    can tell that apart from a PDF that was scanned and came back clean. The
    answer to "is it impossible to hold an unscanned PDF and not know it" is
    that the receipt, not the operator's memory, is what says it was scanned.

    ``--allow-identifying`` exists for the drafting case above. It lowers the
    exit code and nothing else: the hits are still printed, and the receipt
    still records them and marks itself overridden, so the override does not
    travel to the bundle build as a clean verdict.
    """
    clean = True
    for pdf in pdfs:
        report = anonymity_scan.scan_pdf(pdf)
        print("\n".join(report.lines()), flush=True)
        for copy in (pdf, PUBLICATION_DIR / pdf.name):
            if copy.is_file():
                report.target = copy
                receipt = anonymity_scan.write_receipt(report, overridden=allow and not report.clean)
                print(f"receipt {receipt.relative_to(ROOT)}", flush=True)
        report.target = pdf
        clean = clean and report.clean
    if not clean:
        print("", flush=True)
        print("=" * 72, flush=True)
        print("ANONYMITY GATE FAILED. The PDFs above were still written, but one of", flush=True)
        print("them names something a double-blind reviewer must not see, or could", flush=True)
        print("not be scanned at all. Fix the source, rebuild, and do not send or", flush=True)
        print("bundle these files until this gate reports clean.", flush=True)
        print("=" * 72, flush=True)
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper-only", action="store_true")
    group.add_argument("--deck-only", action="store_true")
    parser.add_argument(
        "--allow-identifying",
        action="store_true",
        help="still report anonymity hits, but do not fail the build on them (drafting only)",
    )
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
        build_figures()
        products.append(build_paper())
        products.append(build_supplement())
    if not args.paper_only:
        products.append(build_deck())
    manuscripts: list[Path] = []
    for product in products:
        if product.name in MANUSCRIPTS:
            target = PAPER_DIR / product.name
            shutil.copy2(product, target)
            removed = strip_metadata(target)
            print(f"published {target.relative_to(ROOT)} "
                  f"(document metadata and {removed} embedded metadata entries stripped)", flush=True)
            mirror = PUBLICATION_DIR / product.name
            shutil.copy2(target, mirror)
            print(f"synced {mirror.relative_to(ROOT)}", flush=True)
            manuscripts.append(target)
        else:
            target = PUBLICATION_DIR / product.name
            shutil.copy2(product, target)
            print(f"published {target.relative_to(ROOT)}", flush=True)
    if not check_anonymity(manuscripts, allow=args.allow_identifying):
        # 2 rather than 1, so a caller can tell an anonymity failure apart from
        # a LaTeX failure: here the PDFs exist and are wrong, there they do not
        # exist at all, and the two want different responses.
        return 0 if args.allow_identifying else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
