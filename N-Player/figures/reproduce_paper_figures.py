"""Reproduces the shape of the paper's N-team DSAI figures (Appendix B/C).

Not a test -- a manual visual-review script. It renders AU-frequency
heatmaps from ``theory.stationary.au_frequency`` over the same axes the
paper uses, so a human can eyeball whether the three DSAI zones (compliance
/ dilemma / innovation) show up where ``theory.conditions``' closed-form
boundaries (already unit-tested against the paper's own quoted numbers) say
they should:

- **Early DSAI** (``s`` x ``pr``, small ``W``): cf. Figures S7/S8.
- **Late DSAI** (``pfo`` x ``pr``, large ``W``): cf. Figure S9.
- **Late DSAI across collective-risk gamma**: cf. Figure S12.

Usage::

    python figures/reproduce_paper_figures.py [--out DIR] [--resolution N]

Requires ``matplotlib``, which is not one of this repo's dependencies (see
``ai_race/engine_nplayer/README.md``'s own packaging note for the same kind
of exclusion) -- install it into whatever environment runs this script
manually; it is never imported by anything under ``theory/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same two paths N-Player/conftest.py adds for pytest: this directory's
# parent (for `import theory`) and the repo root (for `import ai_race`, not
# currently needed here but kept for parity with how the rest of N-Player/
# resolves imports).
_N_PLAYER_ROOT = Path(__file__).resolve().parent.parent
for _path in (_N_PLAYER_ROOT, _N_PLAYER_ROOT.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory.stationary import au_frequency

#: Held fixed across every figure: cost, prize, population size, and
#: selection intensity match every one of the paper's N-team captions
#: (Figures S6-S9, S12 all state "c = 1, ... B = 10^4 (or 10000), Z = 100,
#: beta = 0.1"). The benefit ``b`` differs by figure below -- Figures S7/S8
#: (early DSAI) use ``b = 4``; Figures S9/S12 (late DSAI) use ``b = 6``.
COMMON_PARAMS = dict(c=1.0, B=10_000.0, z=100, beta=0.1)


def early_dsai_grid(
    *, n: int, w: float = 100.0, b: float = 4.0, pfo: float = 0.5, resolution: int = 25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AU frequency over (speed ``s``, risk ``pr``) -- cf. Figures S7/S8."""

    s_values = np.linspace(1.05, 5.0, resolution)
    pr_values = np.linspace(0.0, 1.0, resolution)
    grid = np.zeros((resolution, resolution))
    for i, pr in enumerate(pr_values):
        for j, s in enumerate(s_values):
            grid[i, j] = au_frequency(
                n=n, s=float(s), b=b, W=w, pfo=pfo, pr=float(pr), **COMMON_PARAMS
            )
    return s_values, pr_values, grid


def late_dsai_grid(
    *,
    n: int,
    w: float = 1_000_000.0,
    s: float = 1.5,
    b: float = 6.0,
    gamma: float = 0.0,
    resolution: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AU frequency over (found-out ``pfo``, risk ``pr``) -- cf. Figure S9/S12."""

    pfo_values = np.linspace(0.0, 1.0, resolution)
    pr_values = np.linspace(0.0, 1.0, resolution)
    grid = np.zeros((resolution, resolution))
    for i, pr in enumerate(pr_values):
        for j, pfo in enumerate(pfo_values):
            grid[i, j] = au_frequency(
                n=n, s=s, b=b, W=w, pfo=float(pfo), pr=float(pr), gamma=gamma,
                **COMMON_PARAMS,
            )
    return pfo_values, pr_values, grid


def _save_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    mesh = ax.pcolormesh(
        x_values, y_values, grid, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis"
    )
    fig.colorbar(mesh, ax=ax, label="AU frequency")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "output"
    )
    parser.add_argument("--resolution", type=int, default=25)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for n in (3, 5, 10):
        s_values, pr_values, grid = early_dsai_grid(n=n, resolution=args.resolution)
        _save_heatmap(
            s_values,
            pr_values,
            grid,
            xlabel="speed, s",
            ylabel="risk probability, pr",
            title=f"Early DSAI, N={n} (cf. Fig. S7/S8)",
            path=args.out / f"early_dsai_n{n}.png",
        )

    for n in (3, 4, 5):
        pfo_values, pr_values, grid = late_dsai_grid(n=n, resolution=args.resolution)
        _save_heatmap(
            pfo_values,
            pr_values,
            grid,
            xlabel="found-out probability, pfo",
            ylabel="risk probability, pr",
            title=f"Late DSAI, N={n} (cf. Fig. S9)",
            path=args.out / f"late_dsai_n{n}.png",
        )

    for gamma in (0.0, 0.5, 1.0):
        pfo_values, pr_values, grid = late_dsai_grid(
            n=5, gamma=gamma, resolution=args.resolution
        )
        _save_heatmap(
            pfo_values,
            pr_values,
            grid,
            xlabel="found-out probability, pfo",
            ylabel="risk probability, pr",
            title=f"Late DSAI, N=5, gamma={gamma} (cf. Fig. S12)",
            path=args.out / f"late_dsai_n5_gamma{gamma}.png",
        )

    print(f"Wrote {3 + 3 + 3} figures to {args.out}")


if __name__ == "__main__":
    main()
