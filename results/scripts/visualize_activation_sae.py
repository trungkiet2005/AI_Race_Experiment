#!/usr/bin/env python3
"""Validate, merge, and visualize sharded activation-SAE result directories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2563EB"
GOLD = "#D79B00"
INK = "#172033"
GRID = "#D9E0EA"
MUTED = "#6B7280"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete run: {path}")
    for name, expected in manifest.get("artifacts", {}).items():
        artifact = path / name
        if not artifact.is_file() or sha256_file(artifact) != expected["sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact}")
    return {
        "path": path,
        "manifest": manifest,
        "samples": pd.read_csv(path / "samples.csv"),
        "fidelity": pd.read_csv(path / "reconstruction_metrics.csv"),
        "probe": pd.read_csv(path / "probe_metrics.csv"),
        "controls": pd.read_csv(path / "negative_controls.csv"),
        "associations": pd.read_csv(path / "feature_action_associations.csv"),
    }


def _validate_runs(runs: list[dict[str, Any]]) -> None:
    first = runs[0]
    reference = first["samples"].sort_values("sample_id")[
        ["sample_id", "split", "label_unsafe", "rendered_capture_sha256"]
    ].reset_index(drop=True)
    reference_layers: set[int] = set()
    for run in runs:
        manifest = run["manifest"]
        if manifest["preset"]["model_revision"] != first["manifest"]["preset"]["model_revision"]:
            raise ValueError("attribution model revisions differ across lanes")
        if manifest["preset"]["sae_revision"] != first["manifest"]["preset"]["sae_revision"]:
            raise ValueError("SAE revisions differ across lanes")
        if manifest["provenance"] != first["manifest"]["provenance"]:
            raise ValueError("decision-model provenance differs across lanes")
        observed = run["samples"].sort_values("sample_id")[reference.columns].reset_index(drop=True)
        if not observed.equals(reference):
            raise ValueError("sample IDs, labels, splits, or capture hashes differ across lanes")
        layers = {int(value) for value in manifest["capture"]["layers"]}
        overlap = reference_layers & layers
        if overlap:
            raise ValueError(f"duplicate layers across lanes: {sorted(overlap)}")
        reference_layers |= layers


def _style_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save(figure: Any, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_fidelity(fidelity: pd.DataFrame, output_dir: Path, n_samples: int) -> None:
    pivot = fidelity.pivot(index="layer", columns="split", values="normalized_mse").sort_index()
    layers = pivot.index.to_numpy()
    figure, axis = plt.subplots(figsize=(8.2, 4.7))
    for split, color, marker in (("train", BLUE, "o"), ("eval", GOLD, "s")):
        axis.plot(layers, pivot[split], marker=marker, color=color, linewidth=2, label=split.title())
    axis.set_title(
        "Pretrained SAE reconstruction fidelity",
        loc="left",
        color=INK,
        weight="bold",
        pad=30,
    )
    axis.text(0, 1.01, f"Normalized MSE at pre-action resid_post; n={n_samples:,} decisions", transform=axis.transAxes, color=MUTED, fontsize=9)
    axis.set_xlabel("Transformer layer", color=INK)
    axis.set_ylabel("Normalized MSE (lower is better)", color=INK)
    axis.set_xticks(layers)
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)
    _save(figure, output_dir, "sae_fidelity_by_layer")


def plot_probe(probe: pd.DataFrame, controls: pd.DataFrame, output_dir: Path, n_eval: int) -> None:
    probe = probe.sort_values("layer")
    layers = probe["layer"].astype(int).to_numpy()
    null = controls.groupby("layer", as_index=True)["roc_auc"].agg(["mean", "std"])
    figure, axis = plt.subplots(figsize=(8.2, 4.7))
    axis.axhline(0.5, color=INK, linestyle="--", linewidth=1, label="Chance AUC")
    axis.errorbar(
        layers,
        [null.loc[layer, "mean"] for layer in layers],
        yerr=[null.loc[layer, "std"] for layer in layers],
        fmt="s",
        color=GOLD,
        capsize=3,
        label="Shuffled-label control (mean ± SD)",
    )
    axis.plot(layers, probe["roc_auc"], marker="o", linewidth=2.2, color=BLUE, label="SAE code probe")
    axis.set_ylim(0.35, 1.02)
    axis.set_xticks(layers)
    axis.set_xlabel("Transformer layer", color=INK)
    axis.set_ylabel("Grouped-eval ROC-AUC", color=INK)
    axis.set_title(
        "SAFE/UNSAFE information in pretrained SAE codes",
        loc="left",
        color=INK,
        weight="bold",
        pad=30,
    )
    axis.text(0, 1.01, f"Race-group holdout; n={n_eval:,} eval decisions", transform=axis.transAxes, color=MUTED, fontsize=9)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axis(axis)
    _save(figure, output_dir, "sae_probe_by_layer")


def plot_feature_confirmation(associations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    train = associations[associations["split"] == "train"].copy()
    evaluation = associations[associations["split"] == "eval"].copy()
    joined = train.merge(evaluation, on=["layer", "feature_id"], suffixes=("_train", "_eval"))
    joined["abs_train_difference"] = joined["unsafe_minus_safe_train"].abs()
    selected = (
        joined.sort_values(["layer", "abs_train_difference"], ascending=[True, False])
        .groupby("layer", as_index=False, group_keys=False)
        .head(8)
        .copy()
    )
    selected["feature"] = selected.apply(lambda row: f"L{int(row.layer)}·F{int(row.feature_id)}", axis=1)
    selected = selected.sort_values(["layer", "abs_train_difference"], ascending=[False, True])
    y = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(9.2, max(6.0, len(selected) * 0.24)))
    axis.axvline(0, color=INK, linewidth=1)
    axis.scatter(selected["unsafe_minus_safe_train"], y, color=BLUE, s=28, label="Train discovery", zorder=3)
    axis.scatter(selected["unsafe_minus_safe_eval"], y, facecolors="white", edgecolors=GOLD, linewidth=1.5, s=34, label="Eval confirmation", zorder=3)
    for index, row in enumerate(selected.itertuples()):
        axis.plot([row.unsafe_minus_safe_train, row.unsafe_minus_safe_eval], [index, index], color=GRID, linewidth=1, zorder=1)
    axis.set_yticks(y, selected["feature"], fontsize=8)
    axis.set_xlabel("Mean activation difference (UNSAFE − SAFE)", color=INK)
    axis.set_title(
        "Train-discovered SAE features and held-out confirmation",
        loc="left",
        color=INK,
        weight="bold",
        pad=30,
    )
    axis.text(
        0,
        1.005,
        "Top eight overlapping train features per layer; raw SAE units (compare sign/replication, not magnitude across layers)",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=9,
    )
    axis.legend(frameon=False, ncol=2, loc="lower right")
    _style_axis(axis)
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    axis.grid(axis="y", visible=False)
    _save(figure, output_dir, "sae_feature_confirmation")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = [_read_run(Path(value)) for value in args.run_dir]
    _validate_runs(runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = runs[0]["samples"]
    fidelity = pd.concat([run["fidelity"] for run in runs], ignore_index=True).sort_values(["layer", "split"])
    probe = pd.concat([run["probe"] for run in runs], ignore_index=True).sort_values("layer")
    controls = pd.concat([run["controls"] for run in runs], ignore_index=True).sort_values(["layer", "iteration"])
    associations = pd.concat([run["associations"] for run in runs], ignore_index=True)
    selected = plot_feature_confirmation(associations, output_dir)
    plot_fidelity(fidelity, output_dir, len(samples))
    plot_probe(probe, controls, output_dir, int((samples["split"] == "eval").sum()))
    fidelity.to_csv(output_dir / "combined_reconstruction_metrics.csv", index=False)
    probe.to_csv(output_dir / "combined_probe_metrics.csv", index=False)
    controls.to_csv(output_dir / "combined_negative_controls.csv", index=False)
    selected.to_csv(output_dir / "selected_feature_confirmation.csv", index=False)
    manifest = {
        "schema_version": "ai-race.activation-sae.visualization.v1",
        "status": "complete",
        "source_runs": [str(run["path"].resolve()) for run in runs],
        "source_manifests_sha256": [sha256_file(run["path"] / "manifest.json") for run in runs],
        "sample_set_verified_identical": True,
        "n_samples": int(len(samples)),
        "n_train": int((samples["split"] == "train").sum()),
        "n_eval": int((samples["split"] == "eval").sum()),
        "layers": sorted(int(value) for value in probe["layer"].unique()),
        "claim_scope": runs[0]["manifest"]["provenance"]["claim_scope"],
        "figures": ["sae_fidelity_by_layer", "sae_probe_by_layer", "sae_feature_confirmation"],
    }
    (output_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
