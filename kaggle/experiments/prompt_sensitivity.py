# %%
"""Bootstrap the controlled prompt-sensitivity smoke/pilot on Kaggle.

This small kernel entrypoint executes the audited experiment implementation from
the pinned repository Dataset. The default is deliberately the two-repetition
smoke condition; a full pilot is a new kernel version with
``AI_RACE_RUN_PROFILE=prompt_sensitivity_pilot`` after the smoke gates pass.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path


REPO_INPUT_DIR = Path("/kaggle/input/ai-race-experiment")
BASELINE_SCRIPT = REPO_INPUT_DIR / "kaggle" / "experiments" / "baseline.py"

os.environ.setdefault("AI_RACE_RUN_PROFILE", "prompt_sensitivity_smoke")
os.environ.setdefault("AI_RACE_REQUIRED_GPU", "RTX PRO 6000")
os.environ.setdefault("AI_RACE_MIN_GPU_VRAM_GIB", "80")

if not BASELINE_SCRIPT.is_file():
    raise FileNotFoundError(
        f"Pinned repository input is missing {BASELINE_SCRIPT}. Attach the exact "
        "private ai-race-experiment Dataset version used to build this kernel."
    )

runpy.run_path(str(BASELINE_SCRIPT), run_name="__main__")
