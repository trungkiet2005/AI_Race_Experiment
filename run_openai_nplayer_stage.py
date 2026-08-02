"""Chay N config openai_*nplayer*.json SONG SONG (bounded), tu bo qua config/model da completed.

Twin cua run_openai_stage.py (2-player), nhung goi ai_race.engine_nplayer.runner
thay vi ai_race.runner.run_experiment. Cung bai hoc rate-limit: MAX_PARALLEL_JOBS=2
(xem run_openai_stage.py de biet ly do 4 gay 39/88 model-run fail hang loat).
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = REPO / ".venv-kaggle/bin/python"
LOG_DIR = REPO / "results/frontier/openai/_logs/nplayer_stage"

MAX_PARALLEL_JOBS = 2

JOBS = [
    ("ai_race/configs/experiment/openai_baseline_nplayer_n3.json", "results/nplayer/openai/baseline_n3"),
    ("ai_race/configs/experiment/openai_baseline_nplayer_n4.json", "results/nplayer/openai/baseline_n4"),
    ("ai_race/configs/experiment/openai_baseline_nplayer_n5.json", "results/nplayer/openai/baseline_n5"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_AAA_n3.json", "results/nplayer/openai/coopadv/AAA"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_AAC_n3.json", "results/nplayer/openai/coopadv/AAC"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_ACA_n3.json", "results/nplayer/openai/coopadv/ACA"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_ACC_n3.json", "results/nplayer/openai/coopadv/ACC"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_CAA_n3.json", "results/nplayer/openai/coopadv/CAA"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_CAC_n3.json", "results/nplayer/openai/coopadv/CAC"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_CCA_n3.json", "results/nplayer/openai/coopadv/CCA"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_coopadv_CCC_n3.json", "results/nplayer/openai/coopadv/CCC"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_risk_1_n3.json", "results/nplayer/openai/risk/R1"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_risk_2_n3.json", "results/nplayer/openai/risk/R2"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_risk_3_n3.json", "results/nplayer/openai/risk/R3"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_risk_4_n3.json", "results/nplayer/openai/risk/R4"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_risk_5_n3.json", "results/nplayer/openai/risk/R5"),
    ("ai_race/configs/experiment/openai_persona_nplayer_baseline_risk_6_n3.json", "results/nplayer/openai/risk/R6"),
]


def model_slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() or "model"


def is_completed(output_root: Path, model_name: str) -> bool:
    manifest_path = output_root / model_slug(model_name) / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text()).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def run_job(config_rel: str, output_rel: str) -> tuple[str, str, float]:
    config_path = REPO / config_rel
    output_root = REPO / output_rel
    exp = json.loads(config_path.read_text(encoding="utf-8"))
    name = exp["name"]
    models = exp["models"]

    if all(is_completed(output_root, m) for m in models):
        return (name, "SKIP", 0.0)

    started = time.time()
    result = subprocess.run(
        [str(PY), "-m", "ai_race.engine_nplayer.runner", str(config_path), "--output", str(output_root)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    return (name, status, elapsed)


def main() -> int:
    summary: list[tuple[str, str, float]] = []
    print(f"Chay {len(JOBS)} job, toi da {MAX_PARALLEL_JOBS} job song song. Log tung job o {LOG_DIR}/", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as pool:
        futures = {pool.submit(run_job, c, o): (c, o) for c, o in JOBS}
        for future in as_completed(futures):
            config_rel, output_rel = futures[future]
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
