#!/usr/bin/env python3
"""Redraw gallery Figures 04, 05, and 08 from their saved source tables.

This is a narrow, deterministic wrapper around
``analyze_two_player_paper_figures.py``.  It avoids re-running the statistical
bootstrap while still sending the selected figures through the generator's
explicit opaque-white export boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_two_player_paper_figures as analysis


def main() -> None:
    output = ROOT / "results" / "derived" / "two_player_paper_analysis"
    figure_dir = output / "figures"
    table_dir = output / "tables"
    analysis._style()

    analysis._figure_strategy_composition(
        pd.read_csv(table_dir / "fig04_strategy_composition_source.csv"),
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=600,
    )
    analysis._figure_safety_payoff_frontier(
        pd.read_csv(table_dir / "fig05_safety_payoff_frontier_source.csv"),
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=600,
    )
    analysis._draw_matrix_panels(
        pd.read_csv(table_dir / "risk_persona_matrix_rates.csv"),
        models=["GPT-5 nano", "GPT-5.4 nano"],
        title="Complete 6×6 risk-persona surfaces (system-level Unsafe fraction)",
        base=figure_dir / "fig08_gpt_risk_persona_surfaces",
        table_path=table_dir / "fig08_gpt_risk_persona_surfaces_source.csv",
        dpi=600,
        annotate_missing=False,
    )
    print("redrew gallery figures: fig04, fig05, fig08")


if __name__ == "__main__":
    main()
