"""Canonical paths resolved relative to the :mod:`ai_race` package."""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PKG_DIR / "configs"
PROMPTS_DIR = PKG_DIR / "prompts"

REPO_ROOT = PKG_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
REFERENCES_DIR = REPO_ROOT / "references"

