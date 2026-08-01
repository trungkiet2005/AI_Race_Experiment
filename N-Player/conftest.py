"""Makes ``theory`` (this directory) and ``ai_race`` (the repo root) importable
without requiring an editable install, mirroring the repo's own
``pythonpath = ["."]`` pytest setting one level down.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

for path in (_HERE, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
