"""House-style gate for the manuscripts.

Checks the things that are mechanical and have each shipped in a real PDF:

  1. en dash and em dash anywhere in reader-facing text;
  2. a LaTeX double-hyphen or triple-hyphen outside a numeric range, which the
     typesetter renders as one of those dashes;
  3. a prose line that closes without terminal punctuation immediately before
     a structural break, which is how a truncated sentence gets through. The
     rule looks at the ENDING alone, because the next line being a backslash
     command is exactly the case that defeated three earlier checkers;
  4. a stray "exttt" or "extbf", the signature of a backslash eaten by a
     shell heredoc.

Run: python scripts/check_house_style.py [file ...]
Default targets are the AAMAS supplement and the ARR paper.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = [ROOT / "paper" / "supplementary.tex", ROOT / "paper" / "acl" / "main.tex"]

COMMENT = re.compile(r"(?<!\\)%.*$")
# Venue boilerplate is set by the conference, not by us. The date range in the
# class metadata is the publisher's own format and is not prose.
EXEMPT_LINE = re.compile(r"\\acm(Conference|Booking|ISBN|DOI)|\\citation|\\setcopyright")
EN_EM = re.compile(r"[\u2013\u2014]")
# The ban is on the PUNCTUATION dash, the one used where a comma, a colon or a
# full stop belongs. A LaTeX "--" or "---" typesets as one whenever it is set
# off by space on either side. Closed-up forms are ranges (4.6--7.4), compound
# modifiers (model--risk) and paired surnames (Kruskal--Wallis); those are
# conventional typesetting, not the prose habit the rule exists to stop, and a
# rule that flagged them would flag three dozen correct lines and be switched
# off. So: flag a dash with whitespace, or a line end, on either side.
TEX_DASH = re.compile(r"(?:(?<=\s)|^)-{2,3}(?=\s|$)|(?<=\S)-{2,3}(?=\s|$)|(?:(?<=\s)|^)-{2,3}(?=\S)")
EATEN_BACKSLASH = re.compile(r"(?<![\\A-Za-z])(exttt|extbf|extit|emph\b(?!\{))")
STRUCTURAL = re.compile(r"\\(sub)*section\*?\{|\\paragraph\{|\\begin\{(table|figure|itemize|enumerate)")
PROSE_END_OK = re.compile(r"[.!?:;,]\s*$|[-{}\\%&]\s*$|\}\s*$|\\\\\s*$")


def prose(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("%") or s.startswith("\\"):
        return False
    if s.startswith(("&", "\\\\")):
        return False
    return True


def check(path: Path) -> int:
    raw = path.read_text(encoding="utf-8").splitlines()
    body = [COMMENT.sub("", l) for l in raw]
    # the conference metadata spans several lines; blank the whole declaration
    for i, line in enumerate(body):
        if EXEMPT_LINE.search(line):
            for j in range(i, min(i + 3, len(body))):
                body[j] = ""
    bad = 0
    for i, line in enumerate(body, 1):
        for m in EN_EM.finditer(line):
            print(f"{path.name}:{i}: FAIL en/em dash: {line.strip()[:90]}")
            bad += 1
        for m in TEX_DASH.finditer(line):
            print(f"{path.name}:{i}: FAIL LaTeX dash ligature '{m.group(0)}': {line.strip()[:90]}")
            bad += 1
        for m in EATEN_BACKSLASH.finditer(line):
            print(f"{path.name}:{i}: FAIL eaten backslash '{m.group(0)}': {line.strip()[:90]}")
            bad += 1

    for i in range(len(body) - 1):
        line = body[i]
        if not prose(line):
            continue
        nxt = body[i + 1].strip()
        if nxt and not STRUCTURAL.match(nxt):
            continue
        if not nxt and not (i + 2 < len(body) and STRUCTURAL.match(body[i + 2].strip())):
            continue
        if PROSE_END_OK.search(line):
            continue
        print(f"{path.name}:{i+1}: FAIL prose ends without punctuation before a "
              f"structural break: ...{line.strip()[-70:]}")
        bad += 1
    return bad


def main() -> int:
    targets = [Path(a) for a in sys.argv[1:]] or DEFAULT
    total = 0
    for t in targets:
        n = check(t)
        print(f"{t.name}: {n} house-style problem(s)")
        total += n
    print()
    print("house style clean" if total == 0 else f"{total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
