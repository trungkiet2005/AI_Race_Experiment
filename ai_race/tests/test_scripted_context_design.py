from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[2] / "scripts" / "verify_scripted_context_design.py"


def test_scripted_context_design_checks_pass() -> None:
    spec = importlib.util.spec_from_file_location("verify_scripted_context_design", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.run_checks(verbose=False) == []
