"""Hash the complete impact-report, demo, paper, deck, and audit delivery surface."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "impact_upgrade" / "release_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delivery_paths() -> list[Path]:
    impact = ROOT / "results" / "impact_upgrade"
    paths = [
        impact / "analysis_manifest.json",
        impact / "data_quality_audit.json",
        impact / "artifact.json",
        impact / "impact_report.md",
        impact / "impact_report.html",
        impact / "README.md",
        impact / "experiment_gap_audit.md",
        impact / "xai_claim_audit.md",
        ROOT / "docs" / "demos" / "trajectory_lab" / "index.html",
        ROOT / "docs" / "demos" / "trajectory_lab" / "styles.css",
        ROOT / "docs" / "demos" / "trajectory_lab" / "app.js",
        ROOT / "docs" / "demos" / "trajectory_lab" / "data.js",
        ROOT / "docs" / "experiments" / "context_mapping_fully_crossed_protocol.md",
        ROOT / "docs" / "experiments" / "payoff_scale_invariance_protocol.md",
        ROOT / "docs" / "experiments" / "state_scaffold_factorial_protocol.md",
        ROOT / "docs" / "experiments" / "belief_action_coherence_protocol.md",
        ROOT / "docs" / "experiments" / "impact_experiment_program.md",
        ROOT / "ai_race" / "configs" / "experiment" / "context_mapping_fully_crossed.json",
        ROOT / "ai_race" / "configs" / "experiment" / "payoff_scale_invariance.json",
        ROOT / "ai_race" / "configs" / "experiment" / "state_scaffold_factorial.json",
        ROOT / "kaggle" / "experiments" / "greennode_context_mapping_cross.py",
        ROOT / "kaggle" / "experiments" / "greennode_payoff_scale.py",
        ROOT / "kaggle" / "experiments" / "greennode_state_scaffold.py",
        ROOT / "kaggle" / "experiments" / "greennode_scaffold_comprehension.py",
        ROOT / "ai_race" / "audit" / "payoff_scale.py",
        ROOT / "ai_race" / "audit" / "state_scaffold.py",
        ROOT / "ai_race" / "audit" / "scaffold_comprehension.py",
        ROOT / "results" / "scripts" / "followup_analysis_common.py",
        ROOT / "results" / "scripts" / "analyze_payoff_scale_behavior.py",
        ROOT / "results" / "scripts" / "analyze_payoff_scale_contract.py",
        ROOT / "results" / "scripts" / "analyze_state_scaffold_factorial.py",
        ROOT / "results" / "RESULTS_INDEX.md",
        ROOT / "results" / "catalog.csv",
        ROOT / "results" / "catalog.json",
        ROOT / "results" / "migration_manifest.json",
        ROOT / "results" / "visualization_insight_full.md",
        ROOT / "scripts" / "build_publication.py",
        ROOT / "paper" / "main.tex",
        ROOT / "paper" / "refs.bib",
        ROOT / "slides" / "ai_race_research_deck.tex",
        ROOT / "results" / "artifacts" / "publication" / "ai_race_paper.pdf",
        ROOT / "results" / "artifacts" / "publication" / "ai_race_research_deck.pdf",
    ]
    paths.extend(sorted((impact / "data").glob("*")))
    paths.extend(sorted((impact / "figures").glob("*")))
    paths.extend(sorted(impact.glob("demo_*.png")))
    paths.extend(sorted((ROOT / "results" / "derived" / "payoff_scale_contract").glob("*")))
    unique = {path.resolve() for path in paths if path.is_file()}
    return sorted(unique, key=lambda path: path.as_posix().lower())


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    value = result.stdout.strip()
    return value or None


def build_manifest() -> dict:
    files = {}
    for path in delivery_paths():
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {
        "schema_version": "ai-race.impact-release.v1",
        "git_base_revision": git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_value("status", "--porcelain")),
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    current = build_manifest()
    if args.check:
        recorded = json.loads(output.read_text(encoding="utf-8"))
        if recorded != current:
            raise SystemExit("release manifest is stale")
        print(f"verified {current['file_count']} release files")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(current, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"hashed {current['file_count']} release files into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
