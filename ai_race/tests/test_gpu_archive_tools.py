from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "results" / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_audit_rejects_path_traversal() -> None:
    audit = _load_script("audit_gpu_archives.py")
    assert audit.safe_member_name("lane-a/run_manifest.json")
    assert not audit.safe_member_name("../private/key")
    assert not audit.safe_member_name("/absolute/path")


def test_result_path_sanitizer_recurses_without_changing_values() -> None:
    sanitizer = _load_script("sanitize_result_paths.py")
    payload = {"paths": ["private/run/a", "private/run/b"], "n": 3}
    assert sanitizer.replace_strings(payload, "private/run", "archive::run") == {
        "paths": ["archive::run/a", "archive::run/b"],
        "n": 3,
    }
