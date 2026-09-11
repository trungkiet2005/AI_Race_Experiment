"""Which strategy takes the population, at each of the three configured risks.

An invasion diagram is a claim about direction and about nothing else.  Two
things had to be settled before this one could be drawn honestly.

DIRECTION.  EGTtools returns ``fixation_probabilities[i, j]`` meaning the
probability that strategy **j** fixates in a resident population of strategy
**i**.  The archived source in this repository says so in one line,
``tmp/EGTTools-docs-upstream/src/egttools/analytical/sed_analytical.py:719``::

    fp = self.fixation_probability(second, first, beta, *args)
    fixation_probabilities[first, second] = fp

with the signature ``fixation_probability(invader, resident, beta)`` at line
571.  Row is the resident; column is the invader.  The first version of this
script wrote the row out as ``focal_strategy`` and the column as
``opponent_strategy``, which reads as the exact opposite, and the figure built
on those names drew every arrow backwards.  The payoffs settle it independently:
at risk 0.1 the population sits on Always Unsafe 92% of the time, and the cell
that is 1.0 there is row AS, column AU, so the reading has to be "AU takes over
an AS population" and not the reverse.  This script now writes
``resident_strategy`` and ``invader_strategy``, which cannot be read two ways,
and keeps ``focal_strategy``/``opponent_strategy`` as aliases carrying the
meaning their names imply: focal is the invader.

EDGE VALUES.  Every edge that clears neutral drift is at 1.0000 except one at
0.9973, so per-edge labels were three copies of "1.00" stacked over the middle
of the panel where no edge could be assigned to any of them.  The numbers are
kept in ``egt_frontier_invasion.csv`` and the panel states the minimum instead.

WHAT THIS IS NOT.  The edge topology is the EGTtools small-mutation limit; the
node areas are an independently simulated finite-mutation chain, because the
small-mutation transition matrix has more than one absorbing class at these
payoff scales and its stationary eigenvector is therefore not unique.  The two
layers answer different questions and the panel note says so.  Nothing here is
about the language-model routes: no route ever played this population process.

RUNNING IT.  ``--recompute`` re-runs EGTtools, which needs the
archive-compatible environment; EGTtools publishes no wheel for the CPython this
repository runs on, so the default path redraws the archived numerical output
and re-states its hash.  The redraw never invents a fixation probability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = ROOT / "results" / "frontier" / "egt_frontier_comparison_v2"
FIGDIR = ROOT / "figures" / "paper"
STRATEGIES = ("AS", "AU", "CS", "CAS")
RISKS = (0.1, 0.6, 0.9)
POPULATION = 100
BETA = 2.0
DRIFT = 1.0 / POPULATION

# Safe-start strategies on the left, unsafe-start on the right; unconditional on
# top, conditional below.  The square is read the same way as the colours are:
# horizontal position is which action the rule opens with, vertical position is
# whether it ever changes its mind.
POSITION = {
    "AS": np.array([0.15, 0.80]),
    "AU": np.array([0.85, 0.80]),
    "CS": np.array([0.15, 0.22]),
    "CAS": np.array([0.85, 0.22]),
}
# The only two edges that can cross are the diagonals, so they are bowed apart
# rather than left to meet in the middle of the panel.
BOW = {("AS", "CAS"): 0.20, ("CAS", "AS"): 0.20,
       ("CS", "AU"): -0.20, ("AU", "CS"): -0.20}


def style():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import figstyle

    return figstyle


def strategy_colours() -> dict[str, str]:
    """The palette is defined once, next to the other EGT figure."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from build_egt_frontier_insights import strategy_colours as palette

    return palette()


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
    """Run EGTtools.  ``fixation[i, j]`` is j invading a resident i.

    Kept exactly as it was, including the refusal to substitute a local proxy
    for the compiled class, because the archived CSV this script normally
    redraws was produced by this call.
    """
    from egttools.analytical import PairwiseComparison
    from egttools.games import Matrix2PlayerGameHolder

    game = Matrix2PlayerGameHolder(len(STRATEGIES), matrix)
    model = PairwiseComparison(POPULATION, game)
    transition, fixation = model.calculate_transition_and_fixation_matrix_sml(BETA)
    if not np.isfinite(transition).all() or not np.isfinite(fixation).all():
        raise ValueError("EGTtools returned a non-finite transition or fixation matrix")
    eigenvalues, _ = np.linalg.eig(transition.T)
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
        output[risk] = np.array([float(row[f"frequency_{s}_mean"]) for s in STRATEGIES])
    if set(output) != set(RISKS):
        raise ValueError("main-reference finite-mutation summary is incomplete")
    return output


def records_from_egttools(matrices, shares) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for risk in RISKS:
        fixation, multiplicity = fixation_and_sml_diagnostic(matrices[risk])
        stationary = shares[risk]
        for i, resident in enumerate(STRATEGIES):
            for j, invader in enumerate(STRATEGIES):
                records.append(_record(risk, resident, invader,
                                       float(fixation[i, j]), stationary, multiplicity))
    return records


def _record(risk, resident, invader, probability, stationary, multiplicity):
    return {
        "max_private_risk": risk,
        "resident_strategy": resident,
        "invader_strategy": invader,
        # Aliases for the downstream consumer, carrying the meaning their names
        # imply: the focal strategy is the one doing the invading.
        "focal_strategy": invader,
        "opponent_strategy": resident,
        "fixation_probability": probability,
        "stationary_share": float(stationary[STRATEGIES.index(invader)]),
        "above_neutral_drift": int(probability > DRIFT),
        "sml_eigenvalue_one_multiplicity": multiplicity,
        "node_share_source": "finite_mutation_main_reference_chain",
    }


def records_from_archive(path: Path, shares) -> list[dict[str, object]]:
    """Re-read the archived EGTtools output, correcting only the column names.

    A row written before the direction was settled names the resident
    ``focal_strategy``; a row written after it names the resident
    ``resident_strategy``.  Both are read here, so re-running this script on its
    own output does not swap the pairs a second time.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    corrected = "resident_strategy" in (rows[0] if rows else {})
    records: list[dict[str, object]] = []
    for row in rows:
        if corrected:
            resident, invader = row["resident_strategy"], row["invader_strategy"]
        else:
            resident, invader = row["focal_strategy"], row["opponent_strategy"]
        risk = round(float(row["max_private_risk"]), 1)
        records.append(_record(risk, resident, invader,
                               float(row["fixation_probability"]), shares[risk],
                               int(row["sml_eigenvalue_one_multiplicity"])))
    if len(records) != len(RISKS) * len(STRATEGIES) ** 2:
        raise ValueError(f"{path.name} does not hold a full 4x4 at each of {RISKS}")
    return records


def report(records, shares) -> None:
    for risk in RISKS:
        dominant = STRATEGIES[int(np.argmax(shares[risk]))]
        print(f"  risk {risk}: finite-mutation shares "
              + " ".join(f"{s}:{p:.4f}" for s, p in zip(STRATEGIES, shares[risk]))
              + f"; dominant {dominant}")
        for row in records:
            if row["max_private_risk"] != risk or not row["above_neutral_drift"]:
                continue
            if row["resident_strategy"] == row["invader_strategy"]:
                continue
            print(f"    {row['invader_strategy']:>3} invades "
                  f"{row['resident_strategy']:<3} and fixates with probability "
                  f"{row['fixation_probability']:.4f}")


def draw(records, shares) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    S = style()
    colours = strategy_colours()

    # The panels carry no axes, so the only thing that fixes the saved width is
    # where the subplot boxes sit; running them to the canvas edge is what lets
    # ``save`` fit the figure to the column instead of to its longest caption
    # line.  The height is set so an equal-aspect panel is as wide as its box.
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXT, 2.42))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.13, wspace=0.02)
    for axis, risk, letter in zip(axes, RISKS, "abc"):
        stationary = shares[risk]
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.04, 1.04)
        # No equal aspect.  The panel is a schematic, not a metric space, and an
        # aspect lock would leave the drawing floating inside an axes box whose
        # width is what fixes the saved figure width.
        axis.axis("off")

        # Marker area is in points squared, so the radius in points is what a
        # label has to clear and what an arrowhead has to stop short of.  Both
        # are computed from it rather than guessed, which is why no label sits
        # on a node however large the node gets.
        area = {s: 110.0 + 820.0 * float(v) for s, v in zip(STRATEGIES, stationary)}
        radius = {s: float(np.sqrt(a / np.pi)) for s, a in area.items()}

        drawn = [r for r in records
                 if r["max_private_risk"] == risk and r["above_neutral_drift"]
                 and r["resident_strategy"] != r["invader_strategy"]]
        for row in drawn:
            resident = str(row["resident_strategy"])
            invader = str(row["invader_strategy"])
            bow = BOW.get((resident, invader), 0.0)
            # The arrow is the content of the figure: it runs from the
            # population that gets displaced to the strategy that displaces it,
            # and it is drawn in the invader's colour so the direction survives
            # a reader who only looks at the hues.
            axis.add_patch(FancyArrowPatch(
                POSITION[resident], POSITION[invader],
                arrowstyle="-|>", mutation_scale=8.5,
                connectionstyle=f"arc3,rad={bow:.2f}",
                shrinkA=radius[resident] + 1.5, shrinkB=radius[invader] + 3.0,
                linewidth=1.1, color=colours[invader], alpha=0.95,
                joinstyle="miter", zorder=3))

        for strategy in STRATEGIES:
            point = POSITION[strategy]
            axis.scatter([point[0]], [point[1]], s=area[strategy],
                         color=colours[strategy], edgecolor=S.SURFACE,
                         linewidth=0.8, zorder=4)
            # The name goes inside only when the disc can hold it; otherwise it
            # goes beside the disc, clear of its edge.  Nothing is ever printed
            # over a fill it cannot be read against.
            inside = radius[strategy] >= 9.0
            axis.annotate(
                strategy, xy=tuple(point),
                xytext=(0, 0) if inside else (0, radius[strategy] + 2.0),
                textcoords="offset points", ha="center",
                va="center" if inside else "bottom", fontsize=S.FS_NOTE,
                color=S.SURFACE if inside else S.INK, fontweight="bold", zorder=5)
            axis.annotate(
                f"{stationary[STRATEGIES.index(strategy)]:.0%}", xy=tuple(point),
                xytext=(0, -(radius[strategy] + 2.5)), textcoords="offset points",
                ha="center", va="top", fontsize=S.FS_NOTE, color=S.INK_2, zorder=5)

        dominant = STRATEGIES[int(np.argmax(stationary))]
        floor = min(float(r["fixation_probability"]) for r in drawn)
        S.panel(axis, letter,
                rf"$p_r^{{\max}}$={risk:.1f}: {dominant} holds {stationary.max():.0%}")
        axis.annotate(
            f"{len(drawn)} invasions clear drift; smallest $p$={floor:.3f}",
            xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0, -4),
            textcoords="offset points", ha="center", va="top",
            fontsize=S.FS_NOTE, color=S.MUTED, annotation_clip=False)

    S.caption(
        fig,
        r"Arrow: the invader fixates in the resident population with probability "
        r"above neutral drift $1/Z$=0.01, drawn from the displaced population "
        r"toward the strategy that displaces it, in the invader's colour. Disc "
        r"area: share of the independently simulated finite-mutation chain "
        r"($\beta$=2, $\mu$=0.02), because the small-mutation transition has more "
        r"than one absorbing class here and its stationary vector is not unique. "
        r"No language-model route plays this process.",
        y=-0.05,
    )
    written = S.save(fig, "egt_frontier_invasion", figdir=FIGDIR, width=S.TEXT,
                     formats=("pdf", "png", "svg"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--recompute", action="store_true",
                        help="re-run EGTtools instead of redrawing its archived output")
    args = parser.parse_args()
    input_dir = args.input.resolve()
    payoff_path = input_dir / "egt_expected_payoff_matrices.csv"
    summary_path = input_dir / "egt_stationary_summary.csv"
    output_csv = input_dir / "egt_frontier_invasion.csv"
    output_json = input_dir / "egt_frontier_invasion.json"

    shares = load_finite_mutation_shares(summary_path)
    if args.recompute:
        records = records_from_egttools(load_matrices(payoff_path), shares)
        provenance = "egttools.analytical.PairwiseComparison, re-run in this process"
    else:
        records = records_from_archive(output_csv, shares)
        provenance = (f"redrawn from {output_csv.name} as archived, "
                      f"sha256 {sha256(output_csv)}")
    report(records, shares)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output_json.write_text(
        json.dumps(
            {
                "schema_version": "egttools-frontier-invasion-v2",
                "evidence_class": "diagnostic_egttools_run",
                "source_payoff_matrices": str(payoff_path),
                "source_payoff_matrices_sha256": sha256(payoff_path),
                "egttools_version": "0.1.14.2",
                "backend": "egttools.analytical.PairwiseComparison",
                "numerical_provenance": provenance,
                "population_size": POPULATION,
                "beta": BETA,
                "neutral_drift": DRIFT,
                "risks": list(RISKS),
                "column_semantics": {
                    "resident_strategy": "the population being invaded",
                    "invader_strategy": "the single mutant whose fixation probability this is",
                    "focal_strategy": "alias of invader_strategy",
                    "opponent_strategy": "alias of resident_strategy",
                    "correction": (
                        "schema v1 named the EGTtools row focal_strategy and the column "
                        "opponent_strategy. EGTtools returns fixation_probabilities[resident, "
                        "invader] (sed_analytical.py:571,719), so v1's names were reversed and "
                        "any diagram drawn from them points its arrows the wrong way."
                    ),
                },
                "interpretation": (
                    "Edges are EGTtools small-mutation fixation probabilities above neutral "
                    "drift. Node area is the independently estimated finite-mutation "
                    "main-reference chain share, because the EGTtools small-mutation "
                    "transition can have multiple absorbing classes at these payoff scales."
                ),
                "sml_stationary_warning": (
                    "The EGTtools small-mutation stationary eigenvector is not used when "
                    "eigenvalue 1 is non-simple."
                ),
                "node_share_source": summary_path.name,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    written = draw(records, shares)
    for path in written:
        shutil.copy2(path, input_dir / path.name)
    print(json.dumps(
        {"status": "complete",
         "figures": [str(p.relative_to(ROOT)) for p in written],
         "numerical_provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
