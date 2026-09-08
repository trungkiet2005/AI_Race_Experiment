"""Compatibility entry point for rebuilding the repo-level figure hub."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_figure_gallery.py")],
        cwd=ROOT,
        check=True,
    )
