"""Validate and visualize the GreenNode capacity-family admission smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    ROOT / "results" / "capacity_family" / "greennode_smoke_ab0527e"
)
DOMAINS = ("rule_recall", "stage_payoff", "state_update", "terminal_scoring")
DOMAIN_LABELS = ("Rule recall", "Stage payoff", "State update", "Terminal scoring")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(smoke_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    admission_path = smoke_dir / "admission.json"
    manifest_path = smoke_dir / "run_manifest.json"
    raw_path = smoke_dir / "comprehension_raw.jsonl"
    admission = load_json(admission_path)
    manifest = load_json(manifest_path)
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]

    assert manifest["status"] == "completed"
    assert admission["status"] == "complete"
    assert manifest["n_requests"] == manifest["expected_requests"] == 160
    assert admission["coverage"]["passed"] is True
    assert admission["coverage"]["observed_rows"] == len(rows) == 160
    keys = {
        (row["condition_id"], row["mapping_id"], row["repetition"], row["probe_id"])
        for row in rows
    }
    assert len(keys) == 160
    for artifact_name, artifact in manifest["artifacts"].items():
        path = smoke_dir / artifact["path"]
        assert path.stat().st_size == artifact["bytes"], artifact_name
        assert sha256(path) == artifact["sha256"], artifact_name
    assert manifest["model_runtime"]["parameter_devices"] == ["cuda:0"]
    assert manifest["model_runtime"]["parameter_dtypes"] == ["torch.bfloat16"]
    return admission, rows


def summarize(admission: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cells = admission["by_cell"]
    strict_n = sum(cell["strict_parse_n"] for cell in cells.values())
    strict_correct = sum(cell["strict_parse_correct"] for cell in cells.values())
    domain_totals = {
        domain: {
            "n": sum(cell["by_domain"][domain]["n"] for cell in cells.values()),
            "correct": sum(
                cell["by_domain"][domain]["correct"] for cell in cells.values()
            ),
        }
        for domain in DOMAINS
    }
    summary: dict[str, Any] = {
        "model": admission["model"]["short_name"],
        "model_source": admission["model"]["kaggle_model_source"],
        "admission_passed": admission["passed"],
        "n_requests": admission["coverage"]["observed_rows"],
        "strict_parse_correct": strict_correct,
        "strict_parse_n": strict_n,
        "strict_parse_rate": strict_correct / strict_n,
        "arithmetic_checks": sum(cell["arithmetic_checks"] for cell in cells.values()),
        "arithmetic_mismatches": sum(
            cell["arithmetic_mismatches"] for cell in cells.values()
        ),
        "hidden_information_checks": sum(
            cell["hidden_information_checks"] for cell in cells.values()
        ),
        "hidden_information_leaks": sum(
            cell["hidden_information_leaks"] for cell in cells.values()
        ),
        "request_bank_sha256": admission["request_bank_sha256"],
    }
    for domain, totals in domain_totals.items():
        summary[f"{domain}_correct"] = totals["correct"]
        summary[f"{domain}_n"] = totals["n"]
        summary[f"{domain}_accuracy"] = totals["correct"] / totals["n"]

    cell_rows: list[dict[str, Any]] = []
    for cell_id, cell in cells.items():
        for domain in DOMAINS:
            domain_row = cell["by_domain"][domain]
            cell_rows.append(
                {
                    "model": summary["model"],
                    "cell": cell_id,
                    "condition": cell["condition"],
                    "mapping_id": cell["mapping_id"],
                    "domain": domain,
                    "correct": domain_row["correct"],
                    "n": domain_row["n"],
                    "accuracy": domain_row["semantic_accuracy"],
                    "domain_passed": domain_row["passed"],
                    "strict_parse_rate": cell["strict_parse_rate"],
                    "cell_passed": cell["passed"],
                }
            )
    return summary, cell_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figure(output: Path, summaries: list[dict[str, Any]]) -> None:
    metrics = ("strict_parse_rate",) + tuple(f"{domain}_accuracy" for domain in DOMAINS)
    labels = ("Strict parse",) + DOMAIN_LABELS
    values = np.array([[float(row[metric]) for metric in metrics] for row in summaries])
    thresholds = np.array([0.95, 0.80, 0.80, 0.90, 0.90])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    fig.subplots_adjust(left=0.14, right=0.90, bottom=0.16, top=0.78)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(
        range(len(summaries)),
        [row["model"].replace("-instruct", "") for row in summaries],
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            passed = value >= thresholds[column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.1%}\n{'PASS' if passed else 'FAIL'}",
                ha="center",
                va="center",
                color="white" if value >= 0.58 else "#172033",
                fontweight="bold",
            )
            if not passed:
                ax.add_patch(
                    plt.Rectangle(
                        (column_index - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#D99A2B",
                        linewidth=2.5,
                    )
                )
    fig.suptitle(
        "Scaffold admission performance by checkpoint",
        x=0.14,
        y=0.96,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.14,
        0.88,
        "160 unique greedy BF16 probes per model; gold outline marks a failed frozen admission threshold",
        color="#566070",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Accuracy / strict parse rate")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_report(
    path: Path, summaries: list[dict[str, Any]], archive_name: str, archive_sha256: str
) -> None:
    rows = {row["model"]: row for row in summaries}
    q7 = rows["qwen2.5-7b-instruct"]
    q14 = rows["qwen2.5-14b-instruct"]
    lines = [
        "# GreenNode capacity/family admission smoke",
        "",
        "Status: **diagnostic comprehension failure; gameplay not admitted**.",
        "",
        "Every checkpoint below completed 160/160 unique probes with greedy decoding, BF16 parameters, and CUDA-only placement. All failed the frozen admission rule, so no live-game behavior was launched.",
        "",
        "| Model | Strict parse | Rule recall | Stage payoff | State update | Terminal scoring | Admission |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {strict_parse_rate:.1%} | {rule_recall_accuracy:.1%} | "
            "{stage_payoff_accuracy:.1%} | {state_update_accuracy:.1%} | "
            "{terminal_scoring_accuracy:.1%} | **FAIL** |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Main diagnostic",
            "",
            f"Moving from 7B to 14B changed strict parsing from {q7['strict_parse_rate']:.1%} to {q14['strict_parse_rate']:.1%}, but did not solve the substantive bottleneck: state-update accuracy was {q7['state_update_accuracy']:.1%} versus {q14['state_update_accuracy']:.1%}, and terminal scoring was {q7['terminal_scoring_accuracy']:.1%} versus {q14['terminal_scoring_accuracy']:.1%}. Both models were perfect on rule recall and stage payoff. This is a two-checkpoint smoke, not a scaling estimate.",
            "",
            "All disclosed arithmetic-tool outputs matched the engine and no hidden-information leak was detected. That verifies the scaffold interface, not internal game understanding.",
        ]
    )
    mistral = rows.get("mistral-7b-instruct-v0.1")
    if mistral is not None:
        lines.extend(
            [
                "",
                "## Family replication",
                "",
                f"Mistral-7B achieved {mistral['strict_parse_rate']:.1%} strict parsing, {mistral['rule_recall_accuracy']:.1%} rule recall, {mistral['stage_payoff_accuracy']:.1%} stage-payoff accuracy, {mistral['state_update_accuracy']:.1%} state-update accuracy, and {mistral['terminal_scoring_accuracy']:.1%} terminal-scoring accuracy. This adds a checkpoint-template replication, not a family-wide estimate.",
            ]
        )
    lines.extend(
        [
            "",
            "## Provenance and integrity",
            "",
            f"- Source commit: `ab0527eba990dea2620bc03a37b2c33673e58949`",
            f"- Shared request bank: `{q7['request_bank_sha256']}`",
            f"- Download archive SHA-256: `{archive_sha256}`",
            f"- Download archive: `{archive_name}`",
            "- Raw evidence: `results/*/smoke/comprehension_raw.jsonl`",
            "- Runtime receipts: `results/*/smoke/run_manifest.json`",
            "",
            "## Interpretation boundary",
            "",
            "The run has one repetition per scaffold cell and only three named checkpoints: two Qwen sizes and one Mistral checkpoint-template stack. It supports debugging and model admission decisions, not confirmatory model ranking, a universal scaling law, or any claim about gameplay behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--archive-name",
        default="ai_race_capacity_ab0527e_smoke_3models.tar.gz",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    smoke_dirs = sorted((run_root / "results").glob("*/smoke"))
    if len(smoke_dirs) < 2:
        raise RuntimeError(f"Expected at least two smoke directories, found {len(smoke_dirs)}")

    summaries: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for smoke_dir in smoke_dirs:
        admission, _ = validate_run(smoke_dir)
        summary, cells = summarize(admission)
        summaries.append(summary)
        cell_rows.extend(cells)
    order = {
        "qwen2.5-7b-instruct": 0,
        "qwen2.5-14b-instruct": 1,
        "mistral-7b-instruct-v0.1": 2,
    }
    summaries.sort(key=lambda row: order.get(row["model"], 99))
    assert len({row["request_bank_sha256"] for row in summaries}) == 1

    analysis = run_root / "analysis"
    write_csv(analysis / "model_summary.csv", summaries)
    write_csv(analysis / "cell_domain_summary.csv", cell_rows)
    render_figure(analysis / "scaffold_admission_heatmap.png", summaries)
    archive = run_root / args.archive_name
    if not archive.is_file():
        raise FileNotFoundError(f"Missing provenance archive: {archive}")
    write_report(analysis / "README.md", summaries, archive.name, sha256(archive))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
