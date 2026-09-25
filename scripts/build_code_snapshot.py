"""Assemble the anonymous code snapshot that ships inside the supplementary zip.

The paper promises that the code, the de-identified run records and a checker
that recomputes every reported quantity from them come with the supplementary
material. This script makes that true without shipping anything that names the
authors, and writes the result to ``results/_build/code_snapshot/`` (ignored by
git). ``scripts/build_submission_bundle.py`` zips it under ``code/``.

What goes in, and why exactly that:

* ``ai_race/``, the package the experiments ran on;
* ``scripts/verify_manuscript_claims.py``, the modules it imports, and the
  analysers and collection tasks that produced its inputs, with their local
  imports, so every input has its generator beside it;
* exactly the data files the checker opens. They are found by running the
  checker under an audit hook, not by a hand-kept list, so the snapshot cannot
  drift from the checker. The two gitignored derived inputs are among them;
* in place of the source study's participant-level file, a participant-free
  table built from it (``results/derived/human_opening_trajectories``): the
  first-five-round action strings in the source file's order, with each dyad
  renumbered and no participant identifier, age, sex, nationality or elicited
  risk. The builder proves it gives the checker the same arrays;
* the manuscript sources the checker reads, with every comment line and the
  author block removed.

What is left out on purpose: ``docs/`` (except a declaration file the checker
opens), Kaggle metadata JSON, account files and account maps,
``results/public_dataset/``, anything marked DO_NOT_DEPOSIT, ``arr/``,
``paper/acl/``, slides, the web tree, ``vendor/`` and ``.git``.

Every copied text file is sanitised: compute-account handles become stable
pseudonyms (``account_A`` ... in the order of the private account map, so the
labels agree with the supplement), and repository, owner, commit identities,
author emails and absolute local paths become neutral tokens. The identity lists
are read at run time by ``scripts/anonymity_scan.py``; none is written here.
Deep run directories are shortened so no archive path exceeds 150 characters,
and ``PATH_MAP.tsv`` records every move. The snapshot is then scanned file by
file and the build fails if anything identifying is left.

    python scripts/build_code_snapshot.py            # build, scan, run the checker in it
    python scripts/build_code_snapshot.py --no-verify
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import anonymity_scan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "_build" / "code_snapshot"
ARCHIVE_PREFIX = "code/"
MAX_ARCHIVE_PATH = 150
CHECKER = "scripts/verify_manuscript_claims.py"
HUMAN_SOURCE = "references/source_study_dataset/airace_deidentified_long.csv"
HUMAN_DERIVED = "results/derived/human_opening_trajectories/human_opening_trajectories.csv"
ACCOUNT_MAP = ROOT / "docs" / "scripted-opponent-account-map.md"

# The generators of the checker's inputs, and the Kaggle tasks that collected
# the raw run records. Their local imports are added by ``_closure``.
PRODUCERS = [
    "scripts/analyze_trajectory_diversity_rarefaction.py",
    "scripts/analyze_trajectory_diversity_confirmatory.py",
    "scripts/analyze_trajectory_diversity_dyad.py",
    "scripts/analyze_frontier_context_mapping_cross.py",
    "scripts/analyze_nplayer_matched.py",
    "scripts/analyze_scripted_opponent.py",
    "scripts/analyze_baseline_replication.py",
    "scripts/analyze_audit_versus_behaviour.py",
    "scripts/analyze_egt_admitted_routes.py",
    "scripts/analyze_egt_beta_sensitivity.py",
    "scripts/analyze_elicited_risk_by_archetype.py",
    "scripts/analyze_human_archetype_k_sensitivity.py",
    "scripts/analyze_population_identity_grouped.py",
    "scripts/analyze_frontier_admission_campaign.py",
    "scripts/ingest_scripted_cell.py",
    "scripts/ingest_matched_cell.py",
    "results/cross_model_pilot_synthesis/analyze_heterogeneity_test.py",
    "results/cross_model_pilot_synthesis/analyze_feature_importance.py",
    "kaggle/benchmarks/ai_race_baseline.py",
    "kaggle/benchmarks/ai_race_frontier_admission.py",
    "kaggle/benchmarks/ai_race_frontier_context_mapping.py",
    "kaggle/benchmarks/ai_race_nplayer_matched.py",
    "kaggle/benchmarks/ai_race_scripted_opponent.py",
]
IMPORT_DIRS = ["scripts", "scripts/figures", "results/cross_model_pilot_synthesis", "."]
ANALYSIS_PACKAGES = ["numpy", "pandas", "scipy", "matplotlib", "statsmodels", "scikit-learn"]

# Kaggle task output nests every run as <run id>/results/<task>/<route>/, which
# pushes the deepest record past 200 characters. The checker finds these files
# with recursive globs, so collapsing the two redundant levels changes nothing it
# reads. The baseline campaign is left alone: its glob fixes the depth.
COLLAPSE = re.compile(
    r"^(results/frontier/(?:admission_campaign[^/]*|context_mapping_campaign_v3)/.*?/\d+)"
    r"/results/ai_race_[a-z_]+/[^/]+/([^/]+)$")


def excluded(rel: str) -> str | None:
    """Why a path may not ship, or None."""
    low = rel.lower()
    parts = low.split("/")
    name = parts[-1]
    if {".git", "arr", "slides", "web", "vendor", "__pycache__"} & set(parts):
        return "excluded tree"
    if low.startswith("paper/acl/"):
        return "the other paper"
    if low.startswith("results/public_dataset/"):
        return "public dataset staging"
    if "do_not_deposit" in low:
        return "marked DO_NOT_DEPOSIT"
    if "account" in name:
        return "account file"
    if low.startswith("kaggle/") and name.endswith(".json"):
        return "kaggle metadata"
    if low.startswith("references/source_study_dataset/"):
        return "participant-level source data"
    if name.endswith((".pyc", ".pdf", ".zip")):
        return "binary"
    return None


# --------------------------------------------------------------------------- reads
def trace_checker_reads(target: Path) -> None:
    """Run the checker under an audit hook and list every file it opens."""
    opened: set[str] = set()

    def hook(event: str, args: tuple) -> None:
        if event == "open" and args and isinstance(args[0], (str, os.PathLike)):
            opened.add(os.path.abspath(args[0]))

    import runpy

    sys.addaudithook(hook)
    sys.argv = [CHECKER]
    sys.path.insert(0, str(ROOT / "scripts"))
    code = 0
    try:
        runpy.run_path(str(ROOT / CHECKER), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    rel = set()
    for path in opened:
        try:
            rel.add(Path(path).resolve().relative_to(ROOT).as_posix())
        except ValueError:
            continue
    target.write_text("\n".join(sorted(rel)) + "\n", encoding="utf-8")
    sys.exit(code)


def checker_reads() -> tuple[list[str], str]:
    """The files the checker opens, and its verdict line, from a clean run."""
    trace = OUT.parent / "code_snapshot_reads.txt"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_trace", str(trace)],
                          cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    verdict = next((l for l in proc.stdout.splitlines() if "claims verified" in l), "")
    if proc.returncode != 0 or not verdict:
        sys.stdout.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise SystemExit("the checker does not pass on this tree, so there is nothing to snapshot")
    reads = [l for l in trace.read_text(encoding="utf-8").splitlines()
             if l and "__pycache__" not in l and not l.endswith(".pyc")]
    trace.unlink()
    return reads, verdict


# --------------------------------------------------------------------------- code
def _closure(seeds: list[str]) -> set[str]:
    """Every repository module the seeds import, followed transitively."""
    todo, seen = list(seeds), set()
    while todo:
        rel = todo.pop()
        if rel in seen or not (ROOT / rel).is_file():
            continue
        seen.add(rel)
        source = (ROOT / rel).read_text(encoding="utf-8")
        names: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
        for literal in re.findall(r"[\"']([\w./-]+\.py)[\"']", source):
            for base in (ROOT, ROOT / "scripts"):
                if (base / literal).is_file():
                    todo.append((base / literal).resolve().relative_to(ROOT).as_posix())
        for name in names:
            parts = name.split(".")
            for d in IMPORT_DIRS:
                base = ROOT / d
                for k in range(len(parts), 0, -1):
                    cand = base.joinpath(*parts[:k])
                    if cand.with_suffix(".py").is_file():
                        todo.append(cand.with_suffix(".py").relative_to(ROOT).as_posix())
                        break
                    if (cand / "__init__.py").is_file():
                        todo.append((cand / "__init__.py").relative_to(ROOT).as_posix())
                        break
    return seen


def code_files(runtime_modules: list[str]) -> list[str]:
    files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "ai_race").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    files |= _closure([CHECKER, *runtime_modules, *PRODUCERS])
    return sorted(f for f in files if not f.startswith("ai_race/") or (ROOT / f).is_file())


# --------------------------------------------------------------------------- sanitising
def _account_labels(accounts: list[str]) -> dict[str, str]:
    """account_A ... in the order of the private label map, then the rest sorted."""
    order: list[str] = []
    if ACCOUNT_MAP.is_file():
        for label, handle in re.findall(r"\|\s*Account\s+([A-Z])\s*\|\s*`([^`]+)`",
                                        ACCOUNT_MAP.read_text(encoding="utf-8")):
            if handle in accounts and handle not in order:
                order.append(handle)
    order += [a for a in accounts if a not in order]
    labels = {}
    for i, handle in enumerate(order):
        suffix = chr(ord("A") + i) if i < 26 else f"Z{i}"
        labels[handle.lower()] = f"account_{suffix}"
    return labels


REPLACEMENT = {
    "author email": "anonymous@example.org",
    "commit email": "anonymous@example.org",
    "author email name": "anonymous",
    "commit email name": "anonymous",
    "student number": "anonymous-id",
    "github user id": "anonymous-id",
    "institution domain": "example.edu",
    "institution": "ANONYMOUS-INSTITUTION",
    "grant number": "GRANT-ID",
    "repository owner": "anonymous-owner",
    "github user": "anonymous-owner",
    "commit author": "anonymous-author",
    "repository name": "ai-race-artifact",
    "machine name": "anonymous-host",
    "home directory": "anonymous-user",
    "local directory": "anonymous-dir",
}


class Sanitiser:
    def __init__(self) -> None:
        ids = anonymity_scan.load_identities()
        if ids.problems:
            raise SystemExit("cannot sanitise: " + "; ".join(ids.problems))
        self.labels = _account_labels(ids.accounts)
        sep = r"(?:\\\\|\\|/)+"
        # (pattern, replacement, literal the match must contain, lower case)
        self.paths: list[tuple[re.Pattern[str], str, tuple[str, ...]]] = []

        def absolute(path: Path, token: str) -> None:
            parts = [re.escape(p) for p in path.parts[1:]]
            if not parts:
                return
            drive = re.escape(path.drive[:1])
            body = sep.join(parts)
            pattern = rf"(?:{drive}:{sep}|/{drive}/){body}(?![A-Za-z0-9_])"
            self.paths.append((re.compile(pattern, re.I), token, (path.parts[-1].lower(),)))

        absolute(ROOT, "<REPO>")
        for ancestor in ROOT.parents:
            absolute(ancestor, "<LOCAL>")
        homes = {Path.home()}
        short = _short_path(Path.home())
        if short:
            homes.add(short)
        for home in homes:
            absolute(home, "<HOME>")
        for (label, pat), needles in zip(anonymity_scan.GENERIC, anonymity_scan.GENERIC_NEEDLES):
            if label == "home path":
                self.paths.append((pat, "<HOME>", needles))
        self.paths.append((re.compile(r"[A-Za-z0-9._+-]+@users\.noreply\.github\.com", re.I),
                           "anonymous@example.org", ("noreply",)))

        tokens = []
        for token, label in ids.tokens.items():
            if label == "compute account":
                replacement = self.labels[token]
            else:
                replacement = REPLACEMENT.get(label, "ANONYMOUS")
            if anonymity_scan._needs_boundary(token):
                pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
            else:
                pattern = re.escape(token)
            tokens.append((len(token), token, label, re.compile(pattern, re.I), replacement))
        self.tokens = [t[1:] for t in sorted(tokens, key=lambda t: -t[0])]

    def __call__(self, text: str) -> tuple[str, dict[str, int]]:
        counts: dict[str, int] = {}
        for pat, token, needles in self.paths:
            if not any(n in text.lower() for n in needles):
                continue
            text, n = pat.subn(token, text)
            if n:
                counts["local path"] = counts.get("local path", 0) + n
        low = text.lower()
        for literal, label, pat, replacement in self.tokens:
            if literal not in low:
                continue
            text, n = pat.subn(replacement, text)
            low = text.lower()
            if n:
                counts[label] = counts.get(label, 0) + n
        return text, counts


def _short_path(path: Path) -> Path | None:
    """The Windows 8.3 form of a path, which tools write into logs as often."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024):
            return Path(buf.value)
    except (OSError, AttributeError):
        return None
    return None


def strip_manuscript(text: str, name: str) -> str:
    """Drop every comment line and the whole author block.

    Both manuscripts carry the camera-ready author block live in the source,
    hidden only by the anonymous class option.
    """
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("%")]
    start = next((i for i, l in enumerate(lines)
                  if l.lstrip().startswith("\\author")
                  or re.match(r"\s*\\newcommand\{\\\w*[Aa]ffiliation\}", l)), None)
    end = next((i for i, l in enumerate(lines) if "\\begin{abstract}" in l), None)
    if start is None or end is None or start > end:
        raise SystemExit(f"could not locate the author block in {name}")
    lines[start:end] = ["% Author information removed for double-blind review.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- human table
def human_table(dest: Path) -> int:
    """Write the participant-free trajectory table and prove it is equivalent."""
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_trajectory_diversity_rarefaction as td

    human = td.load_human()
    raw = pd.read_csv(td.HUMAN_CSV, usecols=["participant_id", "group_id"])
    group = raw.drop_duplicates("participant_id").set_index("participant_id")["group_id"]
    human = human.assign(group=human["unit"].map(group))
    order = sorted(human["group"].dropna().unique())
    relabel = {g: f"D{i:04d}" for i, g in enumerate(order, start=1)}
    table = pd.DataFrame({
        "risk_cap": human["risk_cap"].astype(float),
        "trajectory": human["trajectory"].astype(str),
        "group": human["group"].map(relabel),
    })
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    back = pd.read_csv(dest, dtype={"trajectory": str, "group": str})

    def arrays(frame):
        cells = [frame[np.isclose(frame["risk_cap"], r)]["trajectory"].tolist() for r in (0.1, 0.6, 0.9)]
        dyads = []
        for r in (0.1, 0.6, 0.9):
            block = frame[np.isclose(frame["risk_cap"], r)]
            dyads.append([g["trajectory"].tolist() for _, g in block.groupby("group") if len(g) == 2])
        return cells, dyads

    if arrays(human) != arrays(back):
        raise SystemExit("the participant-free human table does not reproduce the checker's arrays")
    forbidden = {"participant_id", "group_id", "sex", "age", "nationality_group", "risk_gamble_choice"}
    if forbidden & set(back.columns):
        raise SystemExit("the human table carries a participant-level column")
    return len(back)


# --------------------------------------------------------------------------- project files
def snapshot_pyproject() -> str:
    """The project's pyproject without the vendored package, which does not ship."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    head = text.split("[tool.setuptools]", 1)[0].rstrip() + "\n"
    package_data = re.search(r"ai_race = \[[^\]]*\]", text)
    return head + (
        "\n[tool.setuptools]\n"
        'package-dir = {"" = "."}\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["."]\n'
        'include = ["ai_race*"]\n\n'
        "[tool.setuptools.package-data]\n"
        + (package_data.group(0) if package_data else "") + "\n\n"
        "[tool.pytest.ini_options]\n"
        'pythonpath = ["."]\n'
        'testpaths = ["ai_race/tests"]\n'
        'addopts = "-q"\n'
    )


def snapshot_license() -> str:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    return re.sub(r"(?m)^Copyright \(c\) (\d{4}).*$", r"Copyright (c) \1 Anonymous authors", text)


def lock_file() -> str:
    lines = [f"# Tested with Python {platform.python_version()}. Install with",
             "#   pip install -e .[analysis] -r requirements-lock.txt"]
    for name in ANALYSIS_PACKAGES:
        try:
            lines.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return "\n".join(lines) + "\n"


README = """# Code and run records for the anonymous AAMAS 2027 submission

This archive holds the code, the de-identified run records and a checker that
recomputes every quantity the paper and its supplement report from those
records. It is anonymised for double-blind review and will be released publicly
with the final version.

## Quick check

Python {python} was used to build and test this archive (any Python 3.11 or
later with the pinned packages should work).

```
python -m venv .venv
.venv\\Scripts\\activate            # Windows; on Linux or macOS: source .venv/bin/activate
pip install -e ".[analysis]" -r requirements-lock.txt
python scripts/verify_manuscript_claims.py
```

The last command prints one line per claim and ends with

```
{verdict}
```

It exits with status 0 when every claim holds. A claim that does not match its
artifact is printed as FAIL; a missing input is reported as a failed claim
naming the missing path rather than as a crash. The check takes about a
minute.

On Windows without long-path support, unpack the archive into a short folder
(for example `C:\\snap`): archive paths run to nearly 150 characters, and a file the 260-character limit hides is reported by the checker as a
missing input.

## What is here

| path | contents |
|---|---|
| `ai_race/` | the game engine, prompts, runner and recorder the experiments used |
| `kaggle/benchmarks/` | the hosted-model collection tasks that produced the run records |
| `scripts/verify_manuscript_claims.py` | the checker |
| `scripts/`, `results/cross_model_pilot_synthesis/*.py` | the analysers that produced every derived input the checker reads, and the modules they import |
| `results/frontier/` | raw run records: admission audits, gameplay turns and manifests, scripted-rival cells, crossed context-by-mapping manifests |
| `results/derived/`, `results/open_source/`, `results/cross_model_pilot_synthesis/data/` | the derived tables the checker reads, as the analysers wrote them |
| `results/failed_runs/` | a retained failure record the checker audits |
| `paper/main.tex`, `paper/supplementary.tex` | the manuscript sources, which the checker reads for four claims about the manuscript itself |
| `docs/` | one collection plan declared before its runs, which the checker reads |

The data are exactly the files the checker opens: the archive was assembled by
running the checker under an audit hook and copying what it read.

## Order in which the inputs were produced

The raw records under `results/frontier/` came from the tasks in
`kaggle/benchmarks/`. The derived inputs were then produced by these analysers,
in this order, and the checker was run last:

1. `python scripts/analyze_frontier_admission_campaign.py`
2. `python scripts/analyze_audit_versus_behaviour.py`
3. `python scripts/analyze_scripted_opponent.py`
4. `python scripts/analyze_nplayer_matched.py`
5. `python scripts/analyze_baseline_replication.py`
6. `python scripts/analyze_frontier_context_mapping_cross.py`
7. `python scripts/analyze_trajectory_diversity_rarefaction.py`
8. `python scripts/analyze_trajectory_diversity_confirmatory.py`
9. `python scripts/analyze_trajectory_diversity_dyad.py`
10. `python scripts/analyze_egt_beta_sensitivity.py`, then `python scripts/analyze_egt_admitted_routes.py`
11. `python scripts/analyze_human_archetype_k_sensitivity.py`, `python scripts/analyze_elicited_risk_by_archetype.py`,
    `python scripts/analyze_population_identity_grouped.py`
12. `python results/cross_model_pilot_synthesis/analyze_heterogeneity_test.py --roster five`,
    `python results/cross_model_pilot_synthesis/analyze_feature_importance.py`
13. `python scripts/verify_manuscript_claims.py`

Steps 6, 8 and 9 write the two derived directories
`results/derived/frontier_context_mapping_campaign_v3/` and
`results/derived/trajectory_diversity_confirmatory/`, which are included here
as they were written. Re-running an analyser needs the complete run records
(for example the crossed runs' turn files, or the pilot records behind the
exploratory tables), which will be released with the final version; the checker
itself needs only what is in this archive. The disclosed-arithmetic bootstrap
table (`results/cross_model_pilot_synthesis/data/disclosed_arithmetic_race_bootstrap.csv`)
has no standalone generator; its companion `.json` records the method, seed and
the SHA-256 of the two turn files it was computed from, and the checker
re-reads those turn files for its own claims.

## The human data

The participant-level file of the human study comes from the source study the
paper builds on and is not redistributed here. The checker needs from it only
the first-five-round action strings of each participant and which of them
played together. `results/derived/human_opening_trajectories/` holds exactly
that: {human_rows} rows, one per participant with a complete five-round
opening, in the source file's order, with each dyad renumbered (`D0001`, ...)
and no participant identifier, age, sex, nationality or elicited-risk answer.
With the source file present, the checker reads it directly and the two routes
give identical arrays; the build of this archive checks that.

## Anonymisation

* Compute-account handles are replaced by stable labels (`account_A`,
  `account_B`, ...), so every count of accounts and every "same account"
  comparison is unchanged. Repository, owner and commit identities, author
  emails, institution domains and absolute local paths are replaced by neutral
  tokens (`anonymous-owner`, `<REPO>`, `<HOME>`, ...). `SANITIZED.tsv` lists
  every file whose bytes changed, with the SHA-256 before and after, so the
  released records can be matched to this archive later.
* Deep run directories were shortened so that no archive path exceeds 150
  characters (`PATH_MAP.tsv` maps each moved file to its original location).
  The checker finds these files through recursive patterns, so the move changes
  nothing it reads.
* `SHA256SUMS.txt` lists every other file in the archive; as usual, the checksum
  manifest does not list itself.
"""


# --------------------------------------------------------------------------- build
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-verify", action="store_true",
                        help="do not run the checker inside the finished snapshot")
    parser.add_argument("--_trace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._trace:
        trace_checker_reads(args._trace)
        return 0

    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q",
                              str(OUT.relative_to(ROOT) / "README.md")], check=False)
    if ignored.returncode != 0:
        raise SystemExit(f"{OUT.relative_to(ROOT)} is not ignored by git; refusing to write there")

    print("tracing the files the checker opens ...", flush=True)
    reads, verdict = checker_reads()
    print(f"  {len(reads)} files; repository verdict: {verdict.strip()}", flush=True)

    runtime_modules = [r for r in reads if r.endswith(".py")]
    manuscripts = {"paper/main.tex", "paper/supplementary.tex"}
    data: list[str] = []
    for rel in reads:
        if rel.endswith(".py") or rel in manuscripts or rel == HUMAN_SOURCE:
            continue
        reason = excluded(rel) if not rel.startswith("docs/") else None
        if reason:
            raise SystemExit(f"the checker reads {rel}, which may not ship ({reason})")
        data.append(rel)
    if HUMAN_SOURCE not in reads:
        raise SystemExit("the checker no longer reads the human source file; review the human table")

    code = code_files(runtime_modules)
    for rel in code:
        reason = excluded(rel)
        if reason:
            raise SystemExit(f"{rel} is needed as code but may not ship ({reason})")

    if OUT.exists():
        if OUT.is_symlink() or not OUT.is_dir():
            raise SystemExit(f"refusing to remove non-directory {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    sanitise = Sanitiser()
    path_map: list[tuple[str, str]] = []
    changed: list[str] = []

    def ship(rel: str, *, content: bytes | None = None) -> None:
        dest_rel = rel
        m = COLLAPSE.match(rel)
        if m:
            dest_rel = f"{m.group(1)}/{m.group(2)}"
            path_map.append((dest_rel, rel))
        dest = OUT / dest_rel
        if dest.exists():
            raise SystemExit(f"two files map to {dest_rel}")
        raw = (ROOT / rel).read_bytes() if content is None else content
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            out = raw
        else:
            clean, counts = sanitise(text)
            out = clean.encode("utf-8")
            if counts:
                summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                changed.append(f"{dest_rel}\t{sha256_bytes(raw)}\t{sha256_bytes(out)}\t{summary}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(out)

    for rel in code:
        ship(rel)
    for rel in data:
        ship(rel)
    for rel in sorted(manuscripts):
        text = (ROOT / rel).read_text(encoding="utf-8")
        ship(rel, content=strip_manuscript(text, rel).encode("utf-8"))
    human_rows = human_table(OUT / HUMAN_DERIVED)

    (OUT / "pyproject.toml").write_text(snapshot_pyproject(), encoding="utf-8")
    (OUT / "LICENSE").write_text(snapshot_license(), encoding="utf-8")
    (OUT / "requirements-lock.txt").write_text(lock_file(), encoding="utf-8")
    (OUT / "PATH_MAP.tsv").write_text(
        "shipped_path\toriginal_path\n" + "".join(f"{a}\t{b}\n" for a, b in sorted(path_map)),
        encoding="utf-8")
    (OUT / "SANITIZED.tsv").write_text(
        "path\tsha256_before\tsha256_after\treplacements\n" + "".join(f"{c}\n" for c in sorted(changed)),
        encoding="utf-8")
    (OUT / "README.md").write_text(
        README.format(python=platform.python_version(), verdict=verdict.strip(), human_rows=human_rows),
        encoding="utf-8")

    too_long = [p for p in OUT.rglob("*")
                if p.is_file() and len(ARCHIVE_PREFIX + p.relative_to(OUT).as_posix()) > MAX_ARCHIVE_PATH]
    if too_long:
        raise SystemExit(f"{len(too_long)} archive paths exceed {MAX_ARCHIVE_PATH} characters, "
                         f"e.g. {too_long[0].relative_to(OUT).as_posix()}")
    leaked = [p.relative_to(OUT).as_posix() for p in OUT.rglob("*")
              if p.is_file() and excluded(p.relative_to(OUT).as_posix())
              and not p.relative_to(OUT).as_posix().startswith("docs/")]
    if leaked:
        raise SystemExit(f"excluded material reached the snapshot: {leaked[:5]}")

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_bytes(p.read_bytes())}  {p.relative_to(OUT).as_posix()}\n" for p in files),
        encoding="utf-8")

    print("scanning the snapshot ...", flush=True)
    report = anonymity_scan.scan_tree(OUT)
    print("\n".join(report.lines()[:60]))
    if not report.clean:
        raise SystemExit("the code snapshot is not anonymous; nothing downstream may use it")

    if not args.no_verify:
        print("running the checker inside the snapshot ...", flush=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                   PYTHONPATH=str(OUT))
        proc = subprocess.run([sys.executable, CHECKER], cwd=OUT, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        inside = next((l for l in proc.stdout.splitlines() if "claims verified" in l), "")
        print(f"  {inside.strip() or 'no verdict'} (exit {proc.returncode})")
        if proc.returncode != 0 or inside.strip() != verdict.strip():
            sys.stdout.write("\n".join(l for l in proc.stdout.splitlines() if l.startswith("FAIL")) + "\n")
            sys.stdout.write(proc.stderr[-3000:])
            raise SystemExit("the checker does not reproduce inside the snapshot")

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"snapshot: {len(files) + 1} files, {size / 1048576:.1f} MB raw, "
          f"{len(path_map)} paths shortened, {len(changed)} files sanitised")
    print(f"written to {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
