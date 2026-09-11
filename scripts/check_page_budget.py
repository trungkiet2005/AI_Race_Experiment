"""Refuse a manuscript whose content runs past the venue's page limit.

AAMAS allows at most eight pages of content and an unlimited number of reference
pages, so the quantity that matters is not the page count of the PDF. It is the
page on which the reference list begins. A paper can be nine or ten pages and be
compliant, or be nine pages and be over, and only the position of the
bibliography tells the two apart.

Nothing in this repository was checking it. ``scripts/check_publication.py``
validates fonts, unresolved markers and the anonymous submission identifier, and
never looks at length. That gap is easy to miss precisely because the number it
would report looks fine most of the time: for months the manuscript compiled to
eight pages, but only because the bibliography was empty, since ``bibtex`` was
not being run. Once the citations resolved, the content did not move and the
reference list appeared, and a check on the raw page count would have started
failing a compliant paper while never having caught an over-long one.

So this reads the text layer, finds the page where the reference list starts,
and reports the pages before it as content.

    python scripts/check_page_budget.py
    python scripts/check_page_budget.py --limit 8
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main.pdf"
DEFAULT_LIMIT = 8

# The heading the ACM class emits for the bibliography.  Matched on its own
# line and case-insensitively, so a heading in small caps or full caps is found,
# while the word appearing inside a sentence is not.
HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)


def page_text(pdf: Path, page: int) -> str:
    result = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"pdftotext failed on page {page} of {pdf}")
    return result.stdout


def page_count(pdf: Path) -> int:
    result = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"pdfinfo failed for {pdf}")
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1])
    raise SystemExit(f"no page count in pdfinfo output for {pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PAPER)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"missing PDF: {args.pdf}; compile the manuscript first")

    # A page count read off a stale PDF is worse than no page count, because it
    # is a number and reads like evidence. The submission bundle shipped an
    # eighteen-page supplement this way while every gate reported clean, so the
    # same guard belongs here.
    source = args.pdf.with_suffix(".tex")
    if not source.is_file() and args.pdf.name.startswith("ai_race_"):
        stem = args.pdf.stem.replace("ai_race_", "").replace("paper", "main")
        source = args.pdf.with_name(f"{stem}.tex")
    if source.is_file() and args.pdf.stat().st_mtime < source.stat().st_mtime:
        raise SystemExit(
            f"{args.pdf.name} is older than {source.name}, so this page count "
            "describes a manuscript that no longer exists. Run "
            "scripts/build_publication.py first."
        )

    total = page_count(args.pdf)
    references_page = None
    heading_starts_the_page = False
    for page in range(1, total + 1):
        lines = [line for line in page_text(args.pdf, page).splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if HEADING.match(line):
                references_page = page
                # Whether that page counts against the budget depends on
                # whether anything of the paper is above the heading. A
                # reference list that begins at the top of a page leaves a page
                # of pure references; one that begins halfway down leaves half a
                # page of content, and that page is spent.
                heading_starts_the_page = index == 0
                break
        if references_page is not None:
            break

    if references_page is None:
        # Not a pass.  A manuscript with no reference list is either broken or
        # was compiled without bibtex, and in the second case its page count is
        # smaller than the real one for a reason that has nothing to do with
        # how much has been written.
        raise SystemExit(
            f"{args.pdf.name}: found no reference-list heading in {total} pages. "
            "Either the bibliography is missing or bibtex was not run, and in "
            "both cases the length this reports would be wrong."
        )

    content_pages = references_page - 1 if heading_starts_the_page else references_page
    where = "at the top of" if heading_starts_the_page else "partway down"
    print(f"{args.pdf.name}: {total} pages, references begin {where} page "
          f"{references_page}, so {content_pages} page(s) of content "
          f"against a limit of {args.limit}")

    if content_pages > args.limit:
        raise SystemExit(
            f"content runs to {content_pages} pages, {content_pages - args.limit} "
            f"over the {args.limit}-page limit. Reference pages are unlimited; "
            "content pages are not."
        )
    print("within the page budget")


if __name__ == "__main__":
    main()
