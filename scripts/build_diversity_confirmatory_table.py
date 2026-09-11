"""Emit the supplement's nine-route trajectory-diversity table.

The appendix used to carry only the seven-checkpoint pilot table, which holds
GPT-5 nano (no admission verdict, the endpoint is gone) and holds no row for
GPT-5.4 or GPT-5.5 (two of the five routes the gate admits). A sentence about
"every route that passes the gate" cannot be checked against a table missing
two of them, so the appendix now carries the confirmatory table as well and
this script is what writes it.

Every number comes from
``results/derived/trajectory_diversity_confirmatory/trajectory_diversity_confirmatory.csv``,
written by ``scripts/analyze_trajectory_diversity_confirmatory.py``. Run that
first; this script only formats.

    python scripts/build_diversity_confirmatory_table.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = (ROOT / "results" / "derived" / "trajectory_diversity_confirmatory"
       / "trajectory_diversity_confirmatory.csv")
sys.path.insert(0, str(ROOT / "scripts"))

import figstyle as S  # noqa: E402

RISKS = ("0.1", "0.6", "0.9")


def main() -> None:
    if not CSV.is_file():
        raise SystemExit(
            f"missing {CSV.relative_to(ROOT)}; run "
            "scripts/analyze_trajectory_diversity_confirmatory.py first")
    rows = {}
    with CSV.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[(row["population"], f"{float(row['risk_cap']):.1f}")] = row

    order = ["Human"] + [S.ROUTE_LABEL[route] for route in S.ROUTE_ORDER]
    verdict = {S.ROUTE_LABEL[route]: ("admitted" if route in S.ADMITTED else "refused")
               for route in S.ROUTE_ORDER}

    lines: list[str] = []
    for i, risk in enumerate(RISKS):
        if i:
            lines.append(r"        \midrule")
        for j, population in enumerate(order):
            row = rows[(population, risk)]
            head = risk if j == 0 else ""
            name = population if population == "Human" else \
                f"{population} ({verdict[population]})"
            q0 = (f"{float(row['q0_mean']):.1f} "
                  f"({float(row['q0_ci_low']):.1f}--{float(row['q0_ci_high']):.1f})")
            q1 = (f"{float(row['q1_mean']):.1f} "
                  f"({float(row['q1_ci_low']):.1f}--{float(row['q1_ci_high']):.1f})")
            ham = (f"{float(row['mean_pairwise_hamming']):.3f} "
                   f"({float(row['hamming_ci_low']):.3f}--"
                   f"{float(row['hamming_ci_high']):.3f})")
            lead = f"        {head} & {name}" if head else f"         & {name}"
            lines.append(f"{lead} & {row['source_n']} & {q0} & {q1} & {ham} "
                         f"& {row['n_clusters']} \\\\")
    print("\n".join(lines))

    # The claim the main paper makes, recomputed here so this script fails loudly
    # if the table it prints stops supporting it.
    admitted = [S.ROUTE_LABEL[route] for route in S.ADMITTED]
    below = [
        name for name in [S.ROUTE_LABEL[r] for r in S.ROUTE_ORDER]
        if all(float(rows[(name, risk)]["q1_ci_high"])
               < float(rows[("Human", risk)]["q1_ci_low"]) for risk in RISKS)
    ]
    print(f"\n% every admitted route entirely below the human q1 interval: "
          f"{set(admitted) <= set(below)}", file=sys.stderr)
    print(f"% routes not entirely below: "
          f"{[n for n in [S.ROUTE_LABEL[r] for r in S.ROUTE_ORDER] if n not in below]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
