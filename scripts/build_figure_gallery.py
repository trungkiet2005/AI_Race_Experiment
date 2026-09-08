#!/usr/bin/env python3
"""Build the repo-level figure hub and its deterministic provenance index.

The paper assets and the historical selection gallery are kept in ``figures/``.
This script imports visual outputs from analysis and result directories into a
single diagnostics view without changing the original result artifacts. Exact
duplicates are recorded once and linked to every known source in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
PAPER = FIGURES / "paper"
GALLERY = FIGURES / "gallery"
DIAGNOSTICS = FIGURES / "diagnostics"
MANIFEST = FIGURES / "manifest.json"
INDEX = FIGURES / "INDEX.md"

VISUAL_EXTENSIONS = {".eps", ".jpeg", ".jpg", ".pdf", ".png", ".svg"}
SOURCE_ROOTS = (
    ("open_source", ROOT / "results" / "open_source"),
    ("frontier", ROOT / "results" / "frontier"),
    ("cross_provider", ROOT / "results" / "cross_provider"),
    ("cross_model_pilot_synthesis", ROOT / "results" / "cross_model_pilot_synthesis"),
    ("derived", ROOT / "results" / "derived"),
    ("capacity_family", ROOT / "results" / "capacity_family"),
    ("nplayer", ROOT / "results" / "nplayer"),
    ("visualization_output_archive", ROOT / "results" / "visualization_output_archive"),
    ("impact_upgrade", ROOT / "results" / "impact_upgrade"),
    ("fh_analytic", ROOT / "analysis" / "fh_analytic" / "outputs" / "figures"),
)
EXCLUDED_PARTS = {"_logs", "gemini"}
EXCLUDED_PDF_NAMES = {"manuscript.pdf", "paper.pdf", "report.pdf", "supplementary.pdf", "synthesis_report.pdf"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VISUAL_EXTENSIONS
        and path.name.lower() not in EXCLUDED_PDF_NAMES
        and not EXCLUDED_PARTS.intersection(path.parts)
    )


def posix(path: Path) -> str:
    return path.as_posix()


def gallery_status(path: Path) -> str:
    parts = path.parts
    if any(part.startswith("SELECTED") for part in parts):
        return "selected"
    if any(part.startswith("ADMITTED") for part in parts):
        return "admitted"
    if any(part.startswith("NEW_") for part in parts):
        return "exploratory"
    if "redraw" in parts:
        return "redraw"
    return "gallery"


def add_record(
    records: list[dict],
    seen: dict[str, dict],
    *,
    source: Path,
    canonical: Path,
    kind: str,
    status: str,
) -> None:
    digest = sha256(source)
    record = {
        "canonical_path": posix(canonical.relative_to(ROOT)),
        "kind": kind,
        "sha256": digest,
        "source": posix(source.relative_to(ROOT)),
        "status": status,
    }
    if digest in seen:
        seen[digest].setdefault("sources", []).append(record["source"])
        return
    record["sources"] = [record["source"]]
    seen[digest] = record
    records.append(record)


def import_diagnostics(records: list[dict], seen: dict[str, dict]) -> None:
    if DIAGNOSTICS.exists():
        if DIAGNOSTICS.is_symlink() or not DIAGNOSTICS.is_dir():
            raise SystemExit(f"refusing to replace non-directory: {DIAGNOSTICS}")
        shutil.rmtree(DIAGNOSTICS)
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)

    for label, source_root in SOURCE_ROOTS:
        for source in visual_files(source_root):
            relative = source.relative_to(source_root)
            canonical = DIAGNOSTICS / label / relative
            digest = sha256(source)
            if digest not in seen:
                canonical.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, canonical)
                add_record(
                    records,
                    seen,
                    source=source,
                    canonical=canonical,
                    kind="diagnostic",
                    status="diagnostic",
                )
            else:
                seen[digest].setdefault("sources", []).append(posix(source.relative_to(ROOT)))


def build_index(records: list[dict]) -> str:
    records = sorted(records, key=lambda record: record["canonical_path"])
    paper = [record for record in records if record["kind"] == "paper"]
    gallery = [record for record in records if record["kind"] == "gallery"]
    diagnostics = [record for record in records if record["kind"] == "diagnostic"]

    lines = [
        "# AI Race figure hub",
        "",
        "This is the single browsing entry point for every project-generated figure.",
        "The original result directories remain the provenance source; files here are",
        "canonical paper assets or a hash-tracked selection copy.",
        "",
        f"Total unique visual files: **{len(records)}**",
        "",
        "## How to use this folder",
        "",
        "- Start with `paper/` for figures currently referenced by the manuscript or deck.",
        "- Open `gallery/INDEX.md` for the curated candidate gallery, captions, and evidence labels.",
        "- Use `diagnostics/` for additional result views. Each file is listed below with its source.",
        "- Run `python scripts/build_figure_gallery.py` after generating new figures.",
        "",
        "## Paper and presentation assets",
        "",
        f"{len(paper)} unique files are in the active paper asset set.",
        "",
        "| File | Status | SHA256 |",
        "|---|---|---|",
    ]
    for record in paper:
        link = posix(Path(record["canonical_path"]).relative_to("figures"))
        lines.append(
            f"| [{record['canonical_path'].split('/', 1)[1]}]({link}) | "
            f"{record['status']} | `{record['sha256'][:12]}` |"
        )

    lines.extend(
        [
            "",
            "## Candidate gallery",
            "",
            f"The moved legacy gallery contains {len(gallery)} unique visual files plus its redraw code, tables, and notes.",
            "",
            "- [Open the complete gallery index](gallery/INDEX.md)",
            "- [Open the paper selection note](gallery/SELECTED_FOR_PAPER.md)",
            "- [Open the redraw protocol](gallery/FIGURE_REDRAW_PLAN.md)",
            "",
            "## Diagnostics and additional exports",
            "",
            f"{len(diagnostics)} additional unique visual files are mirrored below.",
            "",
            "| File | Source | SHA256 |",
            "|---|---|---|",
        ]
    )
    for record in diagnostics:
        sources = "<br>".join(f"`{source}`" for source in record["sources"])
        link = posix(Path(record["canonical_path"]).relative_to("figures"))
        lines.append(
            f"| [{record['canonical_path'].split('/', 1)[1]}]({link}) | "
            f"{sources} | `{record['sha256'][:12]}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "A figure's presence in this hub does not promote its evidence class. The source",
            "manifest, admission report, denominator, and protocol still control whether it",
            "can appear in the main paper, supplement, or only as a diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if not PAPER.is_dir() or not GALLERY.is_dir():
        raise SystemExit("expected figures/paper and figures/gallery before syncing")

    records: list[dict] = []
    seen: dict[str, dict] = {}
    for source in visual_files(PAPER):
        add_record(
            records,
            seen,
            source=source,
            canonical=source,
            kind="paper",
            status="paper-ready",
        )
    for source in visual_files(GALLERY):
        add_record(
            records,
            seen,
            source=source,
            canonical=source,
            kind="gallery",
            status=gallery_status(source.relative_to(GALLERY)),
        )

    import_diagnostics(records, seen)
    records.sort(key=lambda record: record["canonical_path"])
    MANIFEST.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    INDEX.write_text(build_index(records), encoding="utf-8")
    print(f"indexed {len(records)} unique visual files")
    print(f"wrote {INDEX.relative_to(ROOT)}")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
