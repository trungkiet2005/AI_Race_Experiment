#!/usr/bin/env python3
"""Recompute N-player scope rates with race-level uncertainty.

The old supplementary table used Wilson intervals with the number of decision
rows as ``n``.  Since decisions within a race are state-dependent, this audit
uses the race as the bootstrap cluster and reports both independent races and
raw decisions for transparency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "results" / "nplayer" / "openai"
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
MODELS = {"gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano"}
SEED = 20260909
N_BOOT = 5000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_cell(n_players: int, model: str, risk: float) -> tuple[pd.DataFrame, Path]:
    path = SOURCE_ROOT / f"baseline_n{n_players}" / model / "turns.jsonl"
    turns = pd.read_json(path, lines=True)
    cell = turns[np.isclose(turns["max_private_risk"].astype(float), risk)].copy()
    if cell.empty:
        raise RuntimeError(f"No turns for N={n_players}, model={model}, risk={risk}")
    if cell["game_id"].nunique() != 10:
        raise RuntimeError(f"Expected 10 independent races in {path} for risk={risk}, found {cell['game_id'].nunique()}")
    return cell, path


def race_bootstrap(cell: pd.DataFrame) -> tuple[float, float, float, int, int]:
    race_rates = cell.groupby("game_id", sort=True)["unsafe"].mean().to_numpy(float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(race_rates, size=(N_BOOT, len(race_rates)), replace=True).mean(axis=1)
    return (
        float(race_rates.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        int(len(race_rates)),
        int(len(cell)),
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    source_hashes = {}
    for n_players in (3, 4, 5):
        for model, label in MODELS.items():
            for risk in (0.1, 0.6, 0.9):
                cell, path = load_cell(n_players, model, risk)
                mean, low, high, n_races, n_decisions = race_bootstrap(cell)
                rows.append({
                    "n_players": n_players, "model": model, "model_label": label,
                    "risk": risk, "unsafe_rate": mean, "ci_low": low, "ci_high": high,
                    "n_races": n_races, "n_decisions": n_decisions,
                })
                source_hashes[str(path.relative_to(ROOT).as_posix())] = sha256(path)
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "nplayer_scope_race_bootstrap.csv", index=False)
    payload = {
        "schema_version": "ai-race-nplayer-scope-race-bootstrap-v1",
        "evidence_class": "exploratory",
        "estimand": "mean player-round Unsafe rate with equal weight per race",
        "ci": "percentile bootstrap over game_id clusters",
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "source_hashes": source_hashes,
        "results": rows,
        "interpretation": "n_races is the independent-unit count; n_decisions is reported only as a descriptive row count.",
    }
    (DATA / "nplayer_scope_race_bootstrap.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print("wrote", DATA / "nplayer_scope_race_bootstrap.csv")
    print("wrote", DATA / "nplayer_scope_race_bootstrap.json")


if __name__ == "__main__":
    main()
