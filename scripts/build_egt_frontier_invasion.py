"""Draw an EGTtools invasion atlas for the reduced AI-race game.

The implementation follows the archive's ``fig_invasion.py`` visual grammar,
but reads only the validated AI-race payoff matrices.  It must be run with the
archive-compatible EGTtools environment because the compiled
``PairwiseComparison`` class is intentionally not replaced by a local proxy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from egttools.analytical import PairwiseComparison
from egttools.games import Matrix2PlayerGameHolder
from egttools.plotting import draw_invasion_diagram


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "frontier" / "egt_frontier_comparison_v2"
STRATEGIES = ("AS", "AU", "CS", "CAS")
RISKS = (0.1, 0.6, 0.9)
POPULATION = 100
BETA = 2.0
DRIFT = 1.0 / POPULATION

INK = "#172033"
MUTED = "#667085"
SURFACE = "#FFFFFF"
STRATEGY_COLORS = {"AS": "#AEB8C8", "AU": "#E78434", "CS": "#3166C6", "CAS": "#C7527A"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrices(path: Path) -> dict[float, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matrices: dict[float, np.ndarray] = {}
    for risk in RISKS:
        matrix = np.array(
            [
                [
                    float(
                        next(
                            row["expected_total_payoff"]
                            for row in rows
                            if float(row["max_private_risk"]) == risk
                            and row["focal_strategy"] == focal
                            and row["opponent_strategy"] == opponent
                        )
                    )
                    for opponent in STRATEGIES
                ]
                for focal in STRATEGIES
            ],
            dtype=float,
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid payoff matrix at risk={risk}")
        matrices[risk] = matrix
    return matrices


def fixation_and_sml_diagnostic(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    game = Matrix2PlayerGameHolder(len(STRATEGIES), matrix)
    model = PairwiseComparison(POPULATION, game)
    transition, fixation = model.calculate_transition_and_fixation_matrix_sml(BETA)
    if not np.isfinite(transition).all() or not np.isfinite(fixation).all():
        raise ValueError("EGTtools returned a non-finite transition or fixation matrix")
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    del eigenvectors
    multiplicity = int(np.sum(np.abs(eigenvalues - 1.0) < 1e-8))
    return fixation, multiplicity


def load_finite_mutation_shares(path: Path) -> dict[float, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[float, np.ndarray] = {}
    for row in rows:
        if row["regime"] != "main_reference":
            continue
        risk = float(row["max_private_risk"])
        output[risk] = np.array([float(row[f"frequency_{strategy}_mean"]) for strategy in STRATEGIES])
    if set(output) != set(RISKS):
        raise ValueError("main-reference finite-mutation summary is incomplete")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    input_dir = args.input.resolve()
    payoff_path = input_dir / "egt_expected_payoff_matrices.csv"
    summary_path = input_dir / "egt_stationary_summary.csv"
    output_pdf = input_dir / "egt_frontier_invasion.pdf"
    output_png = input_dir / "egt_frontier_invasion.png"
    output_csv = input_dir / "egt_frontier_invasion.csv"
    output_json = input_dir / "egt_frontier_invasion.json"
    matrices = load_matrices(payoff_path)
    finite_mutation_shares = load_finite_mutation_shares(summary_path)
    configure_style()

    records: list[dict[str, float | str]] = []
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    figure.subplots_adjust(wspace=0.08, bottom=0.17, top=0.80)
    for axis, risk, panel in zip(axes, RISKS, "abc"):
        fixation, sml_multiplicity = fixation_and_sml_diagnostic(matrices[risk])
        stationary = finite_mutation_shares[risk]
        # The archive encodes stationary mass by node area.  A slightly smaller
        # scale keeps the dominant node and its label inside the three-panel
        # canvas at the largest observed share.
        sizes = [180.0 + 1250.0 * float(value) for value in stationary]
        draw_invasion_diagram(
            list(STRATEGIES),
            DRIFT,
            fixation,
            stationary,
            node_size=sizes,
            font_size_node_labels=8,
            font_size_edge_labels=6,
            font_size_sd_labels=6,
            edge_width=1.4,
            node_linewidth=0.8,
            node_edgecolors=SURFACE,
            max_displayed_label_letters=4,
            colors=[STRATEGY_COLORS[name] for name in STRATEGIES],
            ax=axis,
        )
        axis.set_axis_off()
        axis.text(0.0, 1.06, panel, transform=axis.transAxes, ha="left", va="bottom", fontsize=12, color=INK, fontweight="bold")
        dominant = STRATEGIES[int(np.argmax(stationary))]
        axis.text(
            0.08,
            1.06,
            rf"$r_{{\max}}={risk:.1f}$",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color=INK,
        )
        axis.text(
            0.08,
            1.005,
            f"finite-mutation dominant {dominant} ({stationary.max():.1%})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=MUTED,
        )
        for i, focal in enumerate(STRATEGIES):
            for j, opponent in enumerate(STRATEGIES):
                records.append(
                    {
                        "max_private_risk": risk,
                        "focal_strategy": focal,
                        "opponent_strategy": opponent,
                        "fixation_probability": float(fixation[i, j]),
                        "stationary_share": float(stationary[i]),
                        "above_neutral_drift": int(float(fixation[i, j]) > DRIFT),
                        "sml_eigenvalue_one_multiplicity": sml_multiplicity,
                        "node_share_source": "finite_mutation_main_reference_chain",
                    }
                )
        print(
            f"risk={risk:.1f} finite-mutation node shares="
            + " ".join(f"{s}:{p:.4f}" for s, p in zip(STRATEGIES, stationary))
            + f"; EGTtools SML eigenvalue-1 multiplicity={sml_multiplicity}"
        )

    figure.text(
        0.5,
        0.03,
        r"Node area: finite-mutation main-reference chain ($\beta=2$, $\mu=0.02$); edges: EGTtools PairwiseComparison fixation above neutral drift $1/Z$",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    figure.savefig(output_pdf, facecolor="white")
    figure.savefig(output_png, facecolor="white", dpi=220)
    plt.close(figure)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output_json.write_text(
        json.dumps(
            {
                "schema_version": "egttools-frontier-invasion-v1",
                "evidence_class": "diagnostic_egttools_run",
                "source_payoff_matrices": str(payoff_path),
                "source_payoff_matrices_sha256": sha256(payoff_path),
                "egttools_version": "0.1.14.2",
                "backend": "egttools.analytical.PairwiseComparison",
                "plotter": "egttools.plotting.draw_invasion_diagram",
                "population_size": POPULATION,
                "beta": BETA,
                "neutral_drift": DRIFT,
                "risks": list(RISKS),
                "interpretation": "Edges are EGTtools small-mutation fixation probabilities. Node area is the independently estimated finite-mutation main-reference chain share because the EGTtools small-mutation transition can have multiple absorbing classes at these payoff scales.",
                "sml_stationary_warning": "The EGTtools small-mutation stationary eigenvector is not used when eigenvalue 1 is non-simple.",
                "node_share_source": summary_path.name,
                "outputs": [output_pdf.name, output_png.name, output_csv.name],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_pdf}")
    print(f"wrote {output_csv}")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
