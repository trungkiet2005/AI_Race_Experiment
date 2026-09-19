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

Where the identifying strings come from
---------------------------------------
Version 4 no longer spells any identity out in this file. A list of handles
kept in a public repository is itself a leak: it is the one file that names
every account at once. The strings are loaded at run time instead:

* the compute accounts, from the account pool inventory that lives outside
  this repository (``infra/kaggle_for_research`` in an ancestor directory, or
  the paths in ``ANONYMITY_IDENTITY_SOURCES``);
* the authors, their emails, institutions and grant numbers, from the author
  block and the acknowledgements in ``paper/main.tex``, which the anonymous
  class option hides but which are the camera-ready truth;
* the repository owner, repository name and every commit identity, from git;
* the local directory names above this checkout and the home directory name.

If the account inventory cannot be found the scan still runs, but it records
that it could not check the accounts and returns no verdict, because a scan
that silently checks less is how the original leak passed.

Surfaces
--------
* the text layer, through ``pdftotext``, which is what a reviewer reads;
* the PDF objects, inflated here, with UTF-16 and hex strings decoded, which is
  what a reviewer's PDF viewer reads;
* the metadata of the document and of every embedded form XObject (``/Author``,
  ``/Keywords``, ``/Subject``, ``/Creator``, ``/Title`` and XMP), read through
  pypdf, because an embedded figure carries its own information dictionary and
  one of them once carried a design-tool account ID;
* every file of a directory or archive, for the code snapshot.

Author names are matched only where a name cannot be a citation: metadata, the
first page of a PDF, and code or data files. The source study this paper builds
on was written by two of the authors, and citing it in the third person is
required; a gate that fired on the reference list would be switched off.

Hits are recorded redacted (first three characters), so a receipt checked into
the repository never repeats the string it caught.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

# Bumped whenever the pattern list or the scanned surface changes, so a receipt
# written by an older scanner is not mistaken for a verdict from this one.
SCANNER_VERSION = 4

RECEIPT_SUFFIX = ".anonymity.json"
ROOT = Path(__file__).resolve().parents[1]
IDENTITY_ENV = "ANONYMITY_IDENTITY_SOURCES"

# Patterns that identify by shape rather than by a known value. None of them
# names anybody, so they can live in a public file.
GENERIC: list[tuple[str, re.Pattern[str]]] = [
    # Home-directory paths. A LaTeX console dump, a traceback quoted into an
    # appendix or a "how we ran it" listing carries the machine account name.
    ("home path", re.compile(r"[A-Za-z]:(?:\\\\|\\|/){1,2}Users(?:\\\\|\\|/){1,2}[^\\/\s\"')\]]+", re.I)),
    ("home path", re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+")),
    ("home path", re.compile(r"(?<![A-Za-z0-9])/[a-z]/Users/[^/\s\"']+", re.I)),
    # Personal mail. Restricted to consumer providers: the reference list quotes
    # other people's institutional addresses, and a gate that fires on a correct
    # citation is a gate somebody switches off. Author addresses are loaded.
    ("personal email", re.compile(
        r"[A-Za-z0-9._%+-]+@(?:gmail|googlemail|outlook|hotmail|yahoo|icloud|proton(?:mail)?)\.[A-Za-z.]{2,}", re.I)),
    ("github noreply", re.compile(r"[A-Za-z0-9._+-]+@users\.noreply\.github\.com", re.I)),
    # Credentials. Harmless while the bundle was PDF-only; code now ships.
    ("credential", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("credential", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|ant-[a-z0-9]+-)?[A-Za-z0-9_-]{32,}")),
    ("credential", re.compile(r"gh[pousr]_[A-Za-z0-9]{36}")),
    ("credential", re.compile(r"github_pat_[A-Za-z0-9_]{40,}")),
    ("credential", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{34}(?![A-Za-z0-9])")),
    ("credential", re.compile(r"KGAT_[A-Za-z0-9]{20,}")),
    ("credential", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    ("credential", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # Research-council grant numbers have a fixed shape (for example
    # EP/X00000X/1); the actual numbers are also loaded from the manuscript.
    ("grant number", re.compile(r"(?<![A-Za-z0-9])EP/[A-Z]\d{5}[A-Z0-9]/\d(?![0-9])")),
]
# A literal every match of the pattern at the same index must contain, lower
# case. Checking it first with str.find skips the regex on the many megabytes of
# run records where it cannot match, which is most of the cost of a tree scan.
GENERIC_NEEDLES: list[tuple[str, ...]] = [
    ("users",), ("/home/", "/users/"), ("/users/",),
    ("gmail", "googlemail", "outlook", "hotmail", "yahoo", "icloud", "proton"), ("noreply",),
    ("aiza",), ("sk-",), ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"), ("github_pat_",), ("hf_",), ("kgat_",), ("akia",),
    ("private key",), ("ep/",),
]
assert len(GENERIC_NEEDLES) == len(GENERIC)


def generic_patterns(low: str):
    """The generic patterns that can possibly match a text, given its lower case."""
    for (label, pat), needles in zip(GENERIC, GENERIC_NEEDLES):
        if any(n in low for n in needles):
            yield label, pat


# The ACM template's sample author, left commented in the preamble. It is not a
# person, and flagging it would fire on every copy of the template.
_TEMPLATE_PLACEHOLDERS = {"nimue", "the lady's lake", "lady.of.the.lake@avalon.uk", "avalon", "avalon.uk"}
# Directory names too generic to identify anybody.
_GENERIC_DIRS = {"active", "paused", "archive", "users", "home", "projects", "project",
                 "src", "code", "work", "working", "documents", "desktop", "repos", "git"}
_ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303",
            "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327"}


def fold(text: str) -> str:
    """Lower case, no diacritics, and runs of space or hyphen made one space."""
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    if not text.isascii():
        # Decompose, then drop what is not ASCII: the combining marks go, the
        # base letters stay. Names are matched in the Latin alphabet only.
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[\s\-]+", " ", text.lower())


def _latex_to_text(value: str) -> str:
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\([\'`^\"~=.])\{?([A-Za-z])\}?",
                   lambda m: m.group(2) + _ACCENTS[m.group(1)], value)
    value = re.sub(r"\\([uvHc])\{([A-Za-z])\}", lambda m: m.group(2) + _ACCENTS[m.group(1)], value)
    value = value.replace("{", "").replace("}", "").replace("~", " ")
    return " ".join(unicodedata.normalize("NFC", value).split())


@dataclass
class Identities:
    """What counts as identifying, loaded at run time, never written down here."""

    # literal (lower case) -> category. Matched everywhere.
    tokens: dict[str, str] = field(default_factory=dict)
    # folded name -> category. Matched where a name cannot be a citation.
    people: dict[str, str] = field(default_factory=dict)
    # kaggle handles in a stable order, for pseudonymisation
    accounts: list[str] = field(default_factory=list)
    # where the lists came from, described without a path
    sources: list[str] = field(default_factory=list)
    # why the verdict cannot be trusted, if it cannot
    problems: list[str] = field(default_factory=list)

    def add_token(self, value: str, category: str) -> None:
        value = value.strip().lower()
        if len(value) >= 4 and value not in _TEMPLATE_PLACEHOLDERS:
            self.tokens.setdefault(value, category)

    def add_person(self, value: str, category: str) -> None:
        value = fold(value).strip()
        if len(value) >= 5 and " " in value and value not in _TEMPLATE_PLACEHOLDERS:
            self.people.setdefault(value, category)


def _identity_dirs() -> list[Path]:
    configured = os.environ.get(IDENTITY_ENV)
    if configured:
        return [Path(p) for p in configured.split(os.pathsep) if p]
    found = []
    for ancestor in ROOT.parents:
        candidate = ancestor / "infra" / "kaggle_for_research"
        if candidate.is_dir():
            found.append(candidate)
            break
    return found


def _load_accounts(ids: Identities) -> None:
    handles: set[str] = set()
    for source in _identity_dirs():
        files = [source] if source.is_file() else [source / "pool_inventory.json"]
        for f in files:
            if f.suffix == ".json" and f.is_file():
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                accounts = data.get("accounts") if isinstance(data, dict) else data
                if isinstance(accounts, dict):
                    handles.update(k for k in accounts if isinstance(k, str))
                elif isinstance(accounts, list):
                    handles.update(a.get("username") for a in accounts
                                   if isinstance(a, dict) and a.get("username"))
            elif f.suffix == ".txt" and f.is_file():
                handles.update(line.strip() for line in f.read_text(encoding="utf-8").splitlines()
                               if line.strip() and not line.startswith("#"))
        profiles = source / "runtime_profiles"
        if profiles.is_dir():
            handles.update(p.name for p in profiles.iterdir() if p.is_dir())
    handles = {h for h in handles if re.fullmatch(r"[A-Za-z0-9_.-]{3,}", h)}
    if not handles:
        ids.problems.append(
            "the compute-account inventory was not found (set "
            f"{IDENTITY_ENV} or keep infra/kaggle_for_research beside the projects), "
            "so account handles could not be checked")
        return
    ids.accounts = sorted(handles)
    for h in ids.accounts:
        ids.add_token(h, "compute account")
    ids.sources.append(f"account inventory ({len(handles)} handles)")


def _load_manuscript(ids: Identities, tex: Path) -> None:
    if not tex.is_file():
        ids.problems.append("paper/main.tex is missing, so author names could not be loaded")
        return
    text = tex.read_text(encoding="utf-8", errors="replace")
    preamble = text.split("\\begin{abstract}", 1)[0]
    names = emails = 0
    for m in re.finditer(r"\\author\{([^}]*)\}", preamble):
        name = _latex_to_text(m.group(1))
        ids.add_person(name, "author name")
        names += 1
    for m in re.finditer(r"\\email\{([^}]*)\}", preamble):
        email = m.group(1).strip()
        if email.lower() in _TEMPLATE_PLACEHOLDERS:
            continue
        emails += 1
        ids.add_token(email, "author email")
        local, _, domain = email.partition("@")
        if local.isdigit():
            ids.add_token(local, "student number")
        elif len(local) >= 5:
            ids.add_token(local, "author email name")
        ids.add_token(domain, "institution domain")
        parts = domain.split(".")
        if len(parts) > 2 and parts[-2] in {"edu", "ac"}:
            ids.add_token(".".join(parts[-3:]), "institution domain")
    for m in re.finditer(r"\\institution\{([^}]*)\}", preamble):
        inst = _latex_to_text(m.group(1))
        bare = re.sub(r"\s*\([^)]*\)", "", inst)
        ids.add_person(inst, "institution")
        ids.add_person(bare, "institution")
        # "Centre for X, Y University": the university alone is what a byline
        # would print, and the department alone is too generic to flag.
        for piece in re.split(r",\s*", bare):
            if re.search(r"univ|institut|college", fold(piece)):
                ids.add_person(piece, "institution")
        for acronym in re.findall(r"\(([A-Z][A-Z-]{3,})\)", inst):
            ids.add_token(acronym, "institution")
            ids.add_token(acronym.replace("-", ""), "institution")
    grants = 0
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            for g in re.findall(r"\b[A-Z]{2,}/[A-Z0-9]{5,}/\d+\b", line):
                ids.add_token(g, "grant number")
                grants += 1
    ids.sources.append(f"manuscript author block ({names} names, {emails} emails, {grants} grants)")


def _git(*args: str) -> str:
    if shutil.which("git") is None:
        return ""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=False)
    except OSError:
        return ""
    return out.stdout if out.returncode == 0 else ""


def _load_git(ids: Identities) -> None:
    remotes = _git("remote", "-v")
    for m in re.finditer(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?(?:\s|$)", remotes):
        ids.add_token(m.group(1), "repository owner")
        ids.add_token(m.group(2), "repository name")
    log = _git("log", "--all", "--format=%an%x00%ae%x00%cn%x00%ce")
    people = set()
    for line in log.splitlines():
        people.update(line.split("\x00"))
    for value in people:
        value = value.strip()
        if not value or value.lower() in {"github", "noreply@github.com"}:
            continue
        if "@" in value:
            ids.add_token(value, "commit email")
            local, _, domain = value.partition("@")
            m = re.fullmatch(r"(\d+)\+(.+)", local)
            if m:
                ids.add_token(m.group(1), "github user id")
                ids.add_token(m.group(2), "github user")
            elif len(local) >= 5 and not local.isdigit():
                ids.add_token(local, "commit email name")
            if domain.endswith(".local"):
                ids.add_token(domain[: -len(".local")], "machine name")
        elif " " in value.strip():
            ids.add_person(value, "commit author")
        else:
            ids.add_token(value, "commit author")
    if remotes or log:
        ids.sources.append("git metadata")


def _load_local(ids: Identities) -> None:
    ids.add_token(ROOT.name, "repository name")
    for ancestor in ROOT.parents:
        name = ancestor.name
        if name and name.lower() not in _GENERIC_DIRS and not re.fullmatch(r"[A-Za-z]:?", name):
            ids.add_token(name, "local directory")
    home = Path.home().name
    if home and home.lower() not in _GENERIC_DIRS:
        ids.add_token(home, "home directory")
    for var in ("USERNAME", "USER"):
        if os.environ.get(var):
            ids.add_token(os.environ[var], "home directory")


@lru_cache(maxsize=1)
def load_identities() -> Identities:
    ids = Identities()
    _load_accounts(ids)
    _load_manuscript(ids, ROOT / "paper" / "main.tex")
    _load_git(ids)
    _load_local(ids)
    return ids


def redact(value: str) -> str:
    return value[:3] + "***"


def _scrub(window: str) -> str:
    """Redact every other known identity inside a context window too."""
    for token in load_identities().tokens:
        window = re.sub(re.escape(token), lambda m: redact(m.group(0)), window, flags=re.I)
    for _, pat in generic_patterns(window.lower()):
        window = pat.sub(lambda m: redact(m.group(0)), window)
    return window


def _context(text: str, start: int, end: int) -> str:
    left = _scrub(text[max(0, start - 40):start])
    right = _scrub(text[end:end + 40])
    window = f"{left}{redact(text[start:end])}{right}"
    return " ".join(window.replace("\r", " ").replace("\n", " ").split())


def _bounded(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not before.isalnum() and not after.isalnum()


def _needs_boundary(token: str) -> bool:
    return token.isdigit() or len(token) < 6 or not token.replace(".", "").replace("-", "").isalnum()


MAX_HITS_PER_STRING = 3


def _may_name_people(low: str, ids: Identities) -> bool:
    """Cheap test before folding: does any name's longest word occur at all?

    Folding strips diacritics, so a text that is not pure ASCII is always
    folded; for ASCII text the lower case is already the folded alphabet.
    """
    if not low.isascii():
        return True
    return any(max(name.split(), key=len) in low for name in ids.people)


def scan_text(text: str, origin: str, *, people: bool = True) -> list[str]:
    """Every identifying match in one blob of text, redacted, with context.

    ``people`` switches on author names and institutions. It is off for the
    body of a PDF, whose reference list names the source study's authors.
    """
    ids = load_identities()
    hits: list[str] = []
    low = text.lower()
    for label, pat in generic_patterns(low):
        for n, m in enumerate(pat.finditer(text)):
            if n >= MAX_HITS_PER_STRING:
                break
            hits.append(f"{origin}: {label} -> ...{_context(text, m.start(), m.end())}...")
    for token, label in ids.tokens.items():
        start, n = low.find(token), 0
        while start >= 0 and n < MAX_HITS_PER_STRING:
            end = start + len(token)
            if not _needs_boundary(token) or _bounded(low, start, end):
                hits.append(f"{origin}: {label} {redact(token)} -> ...{_context(text, start, end)}...")
                n += 1
            start = low.find(token, start + 1)
    if people and ids.people and _may_name_people(low, ids):
        folded = fold(text)
        for name, label in ids.people.items():
            start, n = folded.find(name), 0
            while start >= 0 and n < MAX_HITS_PER_STRING:
                end = start + len(name)
                if _bounded(folded, start, end):
                    hits.append(f"{origin}: {label} {redact(name)} -> ...{_context(folded, start, end)}...")
                    n += 1
                start = folded.find(name, start + 1)
    return hits


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
            out.append("    NOT VERIFIED: the artefact could not be fully checked, so it has no verdict")
        elif self.hits:
            out.append(f"    {len(self.hits)} identifying hit(s):")
            out.extend(f"      {h}" for h in self.hits)
        else:
            out.append("    clean")
        return out


def _identity_gate(report: ScanReport) -> bool:
    """Record where the identity lists came from; False if they are incomplete."""
    ids = load_identities()
    report.notes.append("identities loaded from: " + ("; ".join(ids.sources) or "nothing"))
    for problem in ids.problems:
        report.notes.append(problem)
    return not ids.problems


def pdf_text(path: Path, *, first: int | None = None, last: int | None = None) -> str | None:
    """The text layer, or None when ``pdftotext`` is not available to read it."""
    if shutil.which("pdftotext") is None:
        return None
    command = ["pdftotext", "-layout"]
    if first is not None:
        command += ["-f", str(first)]
    if last is not None:
        command += ["-l", str(last)]
    try:
        out = subprocess.run(
            [*command, str(path), "-"],
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
_FONT_MARKERS = (b"eexec", b"%!PS-Adobe", b"/CharStrings", b"/FontMatrix", b"/FontName")


def _is_font_program(payload: bytes) -> bool:
    head = payload[:4096]
    return any(marker in head for marker in _FONT_MARKERS)


_LITERAL = re.compile(rb"\((?:\\.|[^\\()])*\)", re.S)
_HEX = re.compile(rb"<(FEFF[0-9A-Fa-f\s]*)>")
_OCTAL = re.compile(rb"\\([0-7]{1,3}|.)", re.S)
_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}


def _unescape(raw: bytes) -> bytes:
    def sub(m: re.Match[bytes]) -> bytes:
        g = m.group(1)
        if g[:1] in b"01234567":
            return bytes([int(g, 8) & 0xFF])
        return _ESCAPES.get(g, g)
    return _OCTAL.sub(sub, raw)


def decoded_strings(data: bytes) -> list[str]:
    """UTF-16 literal and hex strings, which a latin-1 read turns into noise.

    hyperref writes information strings as ``(\\376\\377\\000M...)`` and other
    producers as ``<FEFF004D...>``; a name in either form evades a byte scan.
    """
    out: list[str] = []
    for m in _LITERAL.finditer(data):
        body = _unescape(m.group(0)[1:-1])
        if body.startswith(b"\xfe\xff"):
            out.append(body[2:].decode("utf-16-be", "replace"))
    for m in _HEX.finditer(data):
        digits = re.sub(rb"\s", b"", m.group(1))
        if len(digits) % 2:
            digits += b"0"
        try:
            out.append(bytes.fromhex(digits[4:].decode()).decode("utf-16-be", "replace"))
        except ValueError:
            continue
    return out


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
    joined = b"\n".join(parts)
    return joined.decode("latin-1", "replace") + "\n" + "\n".join(decoded_strings(joined))


METADATA_KEYS = ("/Author", "/Keywords", "/Subject", "/Creator", "/Title")


def pdf_metadata(path: Path) -> tuple[list[tuple[str, str, str]], int, str | None]:
    """(where, key, value) for the document and every embedded form XObject.

    Also returns how many form XObjects were walked, and an error string
    instead of a result when pypdf cannot read the file.
    """
    try:
        from pypdf import PdfReader
        from pypdf.generic import IndirectObject
    except ImportError:
        return [], 0, "pypdf is not installed, so PDF metadata could not be read"
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - any parse failure means no verdict
        return [], 0, f"pypdf could not read the PDF: {exc}"
    found: list[tuple[str, str, str]] = []

    def info_items(where: str, info) -> None:
        if info is None:
            return
        info = info.get_object()
        for key in METADATA_KEYS:
            if key in info:
                found.append((where, key, str(info[key].get_object())))

    def xmp_items(where: str, holder) -> None:
        meta = holder.get("/Metadata") if holder is not None else None
        if meta is None:
            return
        try:
            found.append((where, "XMP", meta.get_object().get_data().decode("utf-8", "replace")))
        except Exception:  # noqa: BLE001
            found.append((where, "XMP", "<unreadable XMP stream>"))

    info_items("document", reader.trailer.get("/Info"))
    xmp_items("document", reader.trailer["/Root"].get_object())
    seen: set[int] = set()
    forms: list[str] = []

    def walk(resources, page: int) -> None:
        if resources is None:
            return
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            return
        for name, ref in xobjects.get_object().items():
            if isinstance(ref, IndirectObject):
                if ref.idnum in seen:
                    continue
                seen.add(ref.idnum)
            obj = ref.get_object()
            if obj.get("/Subtype") != "/Form":
                continue
            where = f"XObject {name} (page {page})"
            forms.append(where)
            info_items(where, obj.get("/PTEX.InfoDict"))
            xmp_items(where, obj)
            walk(obj.get("/Resources"), page)

    for number, page in enumerate(reader.pages, start=1):
        walk(page.get("/Resources"), number)
    return found, len(forms), None


def scan_pdf(path: Path) -> ScanReport:
    report = ScanReport(path)
    identities_ok = _identity_gate(report)
    text = pdf_text(path)
    first_page = pdf_text(path, first=1, last=1)
    if text is None or first_page is None:
        report.notes.append(
            "pdftotext is not on PATH; install poppler-utils so the text layer "
            "a reviewer reads can be checked"
        )
    else:
        report.verified = True
        report.hits.extend(scan_text(text, f"{path.name} [text layer]", people=False))
        # The byline lives on page one; the reference list, which names the
        # source study's authors, never does.
        report.hits.extend(scan_text(first_page, f"{path.name} [page 1]", people=True))
    try:
        objects = pdf_objects(path)
    except OSError as exc:
        report.notes.append(f"could not read the PDF objects: {exc}")
        report.verified = False
    else:
        report.hits.extend(scan_text(objects, f"{path.name} [pdf objects]", people=False))
        report.notes.append("pdf objects inflated, UTF-16 and hex strings decoded, and scanned")
    metadata, n_forms, problem = pdf_metadata(path)
    if problem:
        report.notes.append(problem)
        report.verified = False
    else:
        for where, key, value in metadata:
            origin = f"{path.name} [{where} {key}]"
            report.hits.extend(scan_text(value, origin, people=True))
            stripped = value.strip().strip("()").strip()
            if key == "/Keywords" and stripped:
                report.hits.append(f"{origin}: metadata is not empty -> {redact(stripped)}")
            if key == "/Author" and stripped and not stripped.lower().startswith("anonymous"):
                report.hits.append(f"{origin}: metadata names an author -> {redact(stripped)}")
        report.notes.append(f"metadata read from the document and {n_forms} embedded form XObjects")
    if not identities_ok:
        report.verified = False
    return report


def _read_text_file(path: Path, origin: str, report: ScanReport) -> None:
    raw = path.read_bytes()
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Not UTF-8 is not the same as not text. Read it byte for byte so an
        # ASCII handle inside a binary or legacy-encoded file is still found.
        body = raw.decode("latin-1")
        report.notes.append(f"{origin}: not UTF-8, scanned as bytes")
    report.hits.extend(scan_text(body, origin, people=True))


def _scan_member(path: Path, origin: str, report: ScanReport) -> None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        inner = scan_pdf(path)
        report.hits.extend(f"{origin} > {h}" for h in inner.hits)
        if not inner.verified:
            report.verified = False
            report.notes.extend(inner.notes)
        return
    if suffix == ".zip":
        inner = scan_zip(path)
        report.hits.extend(f"{origin} > {h}" for h in inner.hits)
        if not inner.verified:
            report.verified = False
        return
    _read_text_file(path, origin, report)


def scan_zip(path: Path) -> ScanReport:
    """Scan an archive member by member.

    A clean PDF sitting beside a dirty archive is still a leak, and the archive
    is the file the portal actually takes.
    """
    report = ScanReport(path)
    identities_ok = _identity_gate(report)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        report.notes.append(f"could not open the archive: {exc}")
        return report
    with archive, tempfile.TemporaryDirectory() as tmp:
        report.verified = True
        members = 0
        for index, name in enumerate(archive.namelist()):
            # A path can identify as readily as its contents: an archive whose
            # members sit under C:/Users/<person>/ needs no reading at all.
            report.hits.extend(scan_text(name, f"{path.name} [member name]", people=True))
            if name.endswith("/"):
                continue
            members += 1
            # Extract under a short numbered name: the member path is already
            # scanned above, and a deep archive path would otherwise overrun
            # the Windows path limit inside the temporary directory.
            target = Path(tmp) / f"m{index}{Path(name).suffix}"
            with archive.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            _scan_member(target, f"{path.name} > {name}", report)
            target.unlink()
        report.notes.append(f"{members} members scanned")
    if not identities_ok:
        report.verified = False
    return report


def scan_tree(path: Path, *, skip: tuple[str, ...] = ("__pycache__",)) -> ScanReport:
    """Scan every file under a directory, and every path name, as a reviewer would."""
    report = ScanReport(path)
    identities_ok = _identity_gate(report)
    if not path.is_dir():
        report.notes.append("not a directory")
        return report
    report.verified = True
    files = 0
    for item in sorted(path.rglob("*")):
        rel = item.relative_to(path).as_posix()
        if any(part in skip for part in item.relative_to(path).parts):
            continue
        report.hits.extend(scan_text(rel, f"{path.name} [path]", people=True))
        if item.is_file():
            files += 1
            _scan_member(item, f"{path.name}/{rel}", report)
    report.notes.append(f"{files} files scanned")
    if not identities_ok:
        report.verified = False
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
    if path.is_dir():
        return scan_tree(path)
    return scan_zip(path) if path.suffix.lower() == ".zip" else scan_pdf(path)


def main(argv: list[str] | None = None) -> int:
    """Scan the artefacts named on the command line. Non-zero if any is not clean."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan PDFs, zips and directories for identifying text.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--receipt", action="store_true", help="write a scan receipt beside each file artefact")
    args = parser.parse_args(argv)

    worst = 0
    for path in args.paths:
        report = scan_artefact(path)
        print("\n".join(report.lines()))
        if args.receipt and path.is_file():
            write_receipt(report)
        if not report.clean:
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
