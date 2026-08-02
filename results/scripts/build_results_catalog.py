"""Build the canonical machine-readable and human-readable results index."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
GENERATED = {
    "catalog.json",
    "catalog.csv",
    "RESULTS_INDEX.md",
    "migration_manifest.json",
}
NON_CATALOG_RECEIPTS = {"release_manifest.json"}
MIGRATED_TREES = (
    "frontier/api_5games_allrisk",
    "open_source/game_understanding_pilot/raw",
    "artifacts/publication",
    "artifacts/qa/latex",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # retained in the quality report
        return None, f"{type(error).__name__}: {error}"
    return (value, None) if isinstance(value, dict) else (None, "root is not an object")


def model_label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("short_name") or value.get("path") or "")
    return str(value or "")


def evidence_class(manifest: dict[str, Any], relative: str) -> str:
    status = str(manifest.get("status") or "unknown")
    if status not in {"completed", "complete", "passed", "admitted"}:
        return (
            "failed-or-incomplete"
            if status in {"failed", "protocol_failed", "incomplete", "aborted"}
            else "unclassified"
        )
    experiment = manifest.get("experiment") or {}
    phase = str(
        manifest.get("run_phase")
        or manifest.get("runPhase")
        or (experiment.get("runPhase") if isinstance(experiment, dict) else "")
        or ""
    ).lower()
    if "confirmatory" in phase:
        return "confirmatory-unadmitted"
    if "pilot" in phase or "pilot" in relative.lower():
        return "pilot"
    if "smoke" in relative.lower() or manifest.get("profile") == "smoke":
        return "diagnostic"
    return "diagnostic"


def manifest_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(RESULTS.rglob("*manifest*.json")):
        if (
            "_build" in path.parts
            or "__pycache__" in path.parts
            or path.name in GENERATED
            or path.name in NON_CATALOG_RECEIPTS
        ):
            continue
        payload, error = safe_json(path)
        relative = path.relative_to(ROOT).as_posix()
        if error:
            errors.append({"path": relative, "error": error})
            continue
        assert payload is not None
        experiment = payload.get("experiment") or {}
        rows.append(
            {
                "path": relative,
                "schema_version": str(payload.get("schema_version") or ""),
                "protocol": str(payload.get("protocol") or payload.get("audit_protocol") or ""),
                "status": str(payload.get("status") or "recorded"),
                "evidence_class": evidence_class(payload, relative),
                "phase": str(
                    payload.get("run_phase")
                    or payload.get("runPhase")
                    or (experiment.get("runPhase") if isinstance(experiment, dict) else "")
                    or ""
                ),
                "model": model_label(payload.get("model")),
                "gpu": str(payload.get("gpu_name") or ""),
                "races": payload.get("n_races", ""),
                "turns": payload.get("n_turns", payload.get("n_outputs", "")),
                "completed_utc": str(payload.get("completed_utc") or ""),
                "sha256": sha256(path),
            }
        )
    return rows, errors


def inventory(hash_all: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in sorted(RESULTS.rglob("*")):
        if (
            not path.is_file()
            or "_build" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or path.name in GENERATED
            or path.name in NON_CATALOG_RECEIPTS
        ):
            continue
        relative = path.relative_to(RESULTS).as_posix()
        digest = sha256(path) if hash_all else ""
        rows.append(
            {
                "path": relative,
                "top_level": relative.split("/", 1)[0],
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        if digest:
            by_hash[digest].append(relative)
    duplicates = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]
    return rows, duplicates


def migration_receipt() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative_tree in MIGRATED_TREES:
        tree = RESULTS / relative_tree
        for path in sorted(tree.rglob("*")) if tree.is_dir() else []:
            if path.is_file():
                relative = path.relative_to(ROOT).as_posix()
                files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {
        "schema_version": "ai-race.results-migration.v1",
        "canonical_root": "results/",
        "retired_roots_absent": {
            "ai_race/results": not (ROOT / "ai_race/results").exists(),
            "output": not (ROOT / "output").exists(),
            "references/output": not (ROOT / "references/output").exists(),
        },
        "migrated_trees": list(MIGRATED_TREES),
        "file_count": len(files),
        "files": files,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(catalog: dict[str, Any], manifests: list[dict[str, Any]]) -> None:
    classes = Counter(row["evidence_class"] for row in manifests)
    visual_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}
    visual_rows = [
        row
        for row in catalog["files"]
        if Path(str(row["path"])).suffix.lower() in visual_suffixes
    ]
    lines = [
        "# AI Race complete results index",
        "",
        "`results/` is the only canonical generated-artifact root. Raw model runs, derived analyses, publication PDFs, QA evidence, and report surfaces live here; source code remains in its normal package directories.",
        "",
        "## At a glance",
        "",
        f"- Canonical files (excluding reproducible `_build/`): **{catalog['file_count']:,}**",
        f"- Total canonical size: **{catalog['total_bytes'] / (1024 ** 2):.2f} MiB**",
        f"- Parsed manifest records: **{len(manifests):,}**",
        f"- Invalid manifest JSON files: **{len(catalog['manifest_errors'])}**",
        f"- Exact duplicate hash groups: **{len(catalog['duplicate_groups'])}** (archives may intentionally retain immutable copies)",
        "",
        "Evidence labels are conservative: a completed pilot stays `pilot`; a prepared or diagnostic artifact is never promoted by directory name alone.",
        "",
        "## Canonical layout",
        "",
        "| Path | Purpose |",
        "|---|---|",
        "| `artifacts/publication/` | Current paper and research-deck PDFs |",
        "| `artifacts/qa/` | Rendered visual QA evidence |",
        "| `open_source/` | Open-weight raw runs, admitted summaries, SAE, EGT, context, and prompt audits |",
        "| `frontier/` | Hosted/frontier model runs, including the migrated API pilot |",
        "| `derived/` | Reproducible analysis outputs |",
        "| `impact_upgrade/` | Cross-study technical report and high-impact visual synthesis |",
        "| `scripts/` | Analysis and catalog builders |",
        "| `_build/` | Ignored, reproducible build scratch |",
        "",
        "## Primary reader entry points",
        "",
        "- [Impact report](impact_upgrade/impact_report.html)",
        "- [Impact report, Markdown](impact_upgrade/impact_report.md)",
        "- [Paper PDF](artifacts/publication/ai_race_paper.pdf)",
        "- [Research deck PDF](artifacts/publication/ai_race_research_deck.pdf)",
        "- [Payoff-scale mechanical contract](derived/payoff_scale_contract/README.md)",
        "- [Experiment impact roadmap](../docs/experiments/impact_experiment_program.md)",
        "",
        f"## Complete visual artifact map ({len(visual_rows):,})",
        "",
        "Every rendered chart, publication PDF, and interactive HTML surface in the canonical results root is linked below. The narrative synthesis embeds the decision-relevant subset in `visualization_insight_full.md`.",
        "",
    ]
    lines.extend(
        f"- [`{row['path']}`](<{row['path']}>)" for row in visual_rows
    )
    lines.extend(
        [
            "",
            "## Evidence ledger counts",
            "",
            "| Evidence class | Manifest records |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(classes.items()))
    lines.extend(
        [
            "",
            "## Run and analysis manifests",
            "",
            "| Manifest | Status | Evidence | Model | Races | Decisions/outputs |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in manifests:
        lines.append(
            "| `{path}` | {status} | {evidence_class} | {model} | {races} | {turns} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Machine-readable receipts",
            "",
            "- `catalog.json`: inventory, directory sizes, manifest audit, and duplicate hashes.",
            "- `catalog.csv`: one row per parsed manifest.",
            "- `migration_manifest.json`: hashes for every file moved from retired result/output roots.",
            "",
        ]
    )
    (RESULTS / "RESULTS_INDEX.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hash-all", action="store_true")
    args = parser.parse_args()
    manifests, manifest_errors = manifest_rows()
    files, duplicates = inventory(hash_all=not args.no_hash_all)
    size_by_top: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in files:
        size_by_top[row["top_level"]]["files"] += 1
        size_by_top[row["top_level"]]["bytes"] += int(row["bytes"])
    catalog = {
        "schema_version": "ai-race.results-catalog.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_root": "results/",
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "size_by_top_level": dict(sorted(size_by_top.items())),
        "manifest_count": len(manifests),
        "manifest_errors": manifest_errors,
        "duplicate_groups": duplicates,
        "files": files,
    }
    (RESULTS / "catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_csv(RESULTS / "catalog.csv", manifests)
    (RESULTS / "migration_manifest.json").write_text(
        json.dumps(migration_receipt(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_markdown(catalog, manifests)
    print(
        json.dumps(
            {
                "files": catalog["file_count"],
                "bytes": catalog["total_bytes"],
                "manifests": len(manifests),
                "manifest_errors": len(manifest_errors),
                "duplicate_groups": len(duplicates),
            },
            indent=2,
        )
    )
    return 1 if manifest_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
