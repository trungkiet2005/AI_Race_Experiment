"""Chay toan bo bedrock_persona_baseline_risk_*.json (ma tran risk-preference
6x6 + risk_averse/risk_seeking) SONG SONG (bounded), tu bo qua config/model
da completed.

Twin cua scripts/run_openai_stage.py, backend Bedrock Converse API (Claude
Opus 5) thay vi OpenAI. Dung cung MAX_PARALLEL_JOBS=2 duoc rut ra tu su co
rate-limit cua openai stage (chay 4 job song song lam 39/88 model-run fail
hang loat) nhu muc mac dinh an toan -- chua co du lieu rate-limit rieng cho
Bedrock de xac nhan hay noi long con so nay.
"""
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv-kaggle/bin/python"
CONFIG_DIR = REPO / "ai_race/configs/experiment"
OUTPUT_ROOT = REPO / "results/frontier/bedrock/persona"
LOG_DIR = REPO / "results/frontier/bedrock/_logs/stage"

MAX_PARALLEL_JOBS = 2

_CELL_LABELS = {"averse": "Rminus_risk_averse", "seeking": "Rplus_risk_seeking"}


def _cell_output(config_name: str) -> Path:
    suffix = config_name[len("bedrock_persona_baseline_risk_"):]
    if suffix in _CELL_LABELS:
        return OUTPUT_ROOT / _CELL_LABELS[suffix]
    i, j = suffix.split("_")
    return OUTPUT_ROOT / "risk_matrix" / f"R{i}_R{j}"


JOBS = [
    (path, _cell_output(path.stem))
    for path in sorted(CONFIG_DIR.glob("bedrock_persona_baseline_risk_*.json"))
]


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() or "model"


def is_completed(output_root: Path, model_name: str) -> bool:
    manifest_path = output_root / model_slug(model_name) / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text()).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def run_job(config_path: Path, output_root: Path) -> tuple[str, str, float]:
    exp = json.loads(config_path.read_text(encoding="utf-8"))
    name = exp["name"]
    models = exp["models"]

    if all(is_completed(output_root, m) for m in models):
        return (name, "SKIP", 0.0)

    started = time.time()
    result = subprocess.run(
        [str(PY), "-m", "ai_race.runner.run_experiment", str(config_path), "--output", str(output_root)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    return (name, status, elapsed)


def main() -> int:
    summary: list[tuple[str, str, float]] = []
    print(f"Chay {len(JOBS)} job, toi da {MAX_PARALLEL_JOBS} job song song. Log tung job o {LOG_DIR}/", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as pool:
        futures = {pool.submit(run_job, c, o): (c, o) for c, o in JOBS}
        for future in as_completed(futures):
            config_path, output_root = futures[future]
            name, status, elapsed = future.result()
            summary.append((name, status, elapsed))
            print(
                f"[{len(summary)}/{len(JOBS)}] {name}: {status} sau {elapsed / 60:.1f} phut",
                flush=True,
            )

    print(f"\n{'=' * 70}\nTONG KET STAGE (song song, max {MAX_PARALLEL_JOBS} job)\n{'=' * 70}")
    for name, status, elapsed in summary:
        print(f"{status:20s} {elapsed / 60:5.1f} phut  {name}")

    failed = [n for n, s, _ in summary if s.startswith("FAIL")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
