# Archive EGTtools execution record

Evidence class: executable archive reference, not an AI Race result.

This directory contains the output of running the archive's original
`Analysis/scaling/s09_egt.py` without editing that archive. The command used
the archive-compatible environment at `D:\Anaconda\envs\egt\python.exe` and
redirected `PD_SCALING_TABLES` and `PD_PAPER_DIR` here. The source script was
read from:

`D:\PhD_LetGoo\PhD_Farming\archive\prisoner-dilemma-llm\Analysis\scaling\s09_egt.py`

Source provenance:

- Archive commit: `176aa933612d361d799ce8b1fd1919f53ad532f1`
- `s09_egt.py` SHA256: `47D0C943CA1979623AC43034D2B4094EE75714EFF5FDB717B4333B2AB0E21200`
- EGTtools version reported by the run: `0.1.14.2`
- Numerical precision: mpmath, 200 decimal digits
- Population: `Z=100`, Fermi `beta=0.1`, ten rounds, execution error `epsilon=0.05` for the comparison table

The archive script completed its self checks and wrote `T14`, `T15`, and the
automatically generated LaTeX table. `T13_pooled_strategy.csv` and
`T11_rule_distance.csv` are staged copies of the archive inputs required by
the script when its output directory is redirected. They are not AI Race
measurements.

The run reproduces the archive's numerical warning at large payoff scale:
float64 has two absorbing states at `lambda=100` and `lambda=1000` in the
`epsilon=0.05` cells, while the 200 digit path assigns the stationary mass to
ALLD. This is retained as a diagnostic rather than silently removed.
