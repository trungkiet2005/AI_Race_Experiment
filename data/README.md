# Data Index

This directory is a copied analysis-ready view of selected raw experiment outputs.
The original files remain under `results/`; nothing here is moved or cut from the
source tree.

## Layout

```text
data/
  experiments/
    players_2/
      provider_<provider>/
        family_<model_family>/
          persona_none/
            mode_baseline/
              condition_nonpersona/
                <model>/
                  raw/
          persona_risk_aware/
            mode_risk_matrix/
              condition_<condition>/
                <model>/
                  raw/
                _condition_raw/
            mode_strategy_persona/
              condition_<condition>/
                <model>/
                  raw/
                _condition_raw/
```

Use `players_2` for current two-player experiments. Future n-player data can use
parallel roots such as `players_3`, `players_5`, or `players_n` without changing
the provider/persona/mode/condition/model convention.

## Naming Contract

- `provider_google/family_gemini`: Gemini-family frontier runs.
- `provider_openai/family_chatgpt`: ChatGPT/OpenAI frontier runs.
- `persona_none`: non-persona baseline runs.
- `persona_risk_aware`: runs with risk-aware or strategy/persona framing.
- `mode_baseline`: non-persona baseline.
- `mode_risk_matrix`: pairwise risk preference matrix, e.g. `condition_R1_R6`.
- `mode_strategy_persona`: named strategy/persona conditions, e.g.
  `condition_S_CC_coop_coop`, `condition_Rplus_risk_seeking`.
- `<model>/raw`: copied per-model raw files such as `all_results.csv`,
  `players.csv`, `races.csv`, `turns.jsonl`, and `run_manifest.json`.
- `_condition_raw`: copied files that were stored at condition level in the
  source tree, usually aggregate `all_results.csv` files.

## Source Mapping

```text
results/frontier/baseline
-> data/experiments/players_2/provider_google/family_gemini/persona_none/mode_baseline/condition_nonpersona

results/frontier/persona/R*_R*_risk_matrix
-> data/experiments/players_2/provider_google/family_gemini/persona_risk_aware/mode_risk_matrix

results/frontier/persona/S_*, R0_neutral, Rminus_risk_averse, Rplus_risk_seeking
-> data/experiments/players_2/provider_google/family_gemini/persona_risk_aware/mode_strategy_persona

results/frontier/openai/baseline
-> data/experiments/players_2/provider_openai/family_chatgpt/persona_none/mode_baseline/condition_nonpersona

results/frontier/openai/persona/risk_matrix
-> data/experiments/players_2/provider_openai/family_chatgpt/persona_risk_aware/mode_risk_matrix

results/frontier/openai/persona/S_*, R0_neutral, Rminus_risk_averse, Rplus_risk_seeking
-> data/experiments/players_2/provider_openai/family_chatgpt/persona_risk_aware/mode_strategy_persona
```
