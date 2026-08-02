"""Chay 6 config con thieu cua ma tran 44-cell (baseline, R0_neutral,
S_CC_coop_coop, S_AA_adv_adv, S_AC_adv_coop, S_CA_coop_adv) cho Claude Opus 5
+ Sonnet 5 qua Bedrock Converse. Twin cua scripts/run_bedrock_stage.py
(36+2 risk cell).

CHI chay sau khi scripts/run_bedrock_stage.py (38-cell risk matrix) da chay
xong -- khong chay dong thoi voi no de tranh gap doi tai dong thoi len
bedrock-runtime (2 job x concurrency 4 = 8 moi stage, cong lai 16 se vuot
muc da xac nhan an toan).
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
OUTPUT_ROOT = REPO / "results/frontier/bedrock"
LOG_DIR = REPO / "results/frontier/bedrock/_logs/stage"

MAX_PARALLEL_JOBS = 2

JOBS_MAP = {
    "bedrock_baseline": OUTPUT_ROOT / "baseline",
    "bedrock_persona_baseline_neutral": OUTPUT_ROOT / "persona" / "R0_neutral",
    "bedrock_persona_baseline_coop_coop": OUTPUT_ROOT / "persona" / "S_CC_coop_coop",
    "bedrock_persona_baseline_adv_adv": OUTPUT_ROOT / "persona" / "S_AA_adv_adv",
    "bedrock_persona_baseline_adv_coop": OUTPUT_ROOT / "persona" / "S_AC_adv_coop",
    "bedrock_persona_baseline_coop_adv": OUTPUT_ROOT / "persona" / "S_CA_coop_adv",
}

JOBS = [(CONFIG_DIR / f"{name}.json", out) for name, out in JOBS_MAP.items()]


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
