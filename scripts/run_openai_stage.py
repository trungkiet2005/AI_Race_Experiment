"""Chay N config openai_*.json SONG SONG (bounded), tu bo qua config/model da completed.

Cac job ghi vao thu muc output khac nhau nen chay dong thoi an toan (khong
dung chung file). Rate limit da xac nhan: 500 request/phut MOI MODEL, nhung
GIOI HAN NAY TINH THEO TO CHUC (org), khong phai theo tien trinh rieng le.

BAI HOC TU LAN CHAY THU: MAX_PARALLEL_JOBS=4 (x concurrency=4/job =
toi da 16 request dong thoi) da lam GAN NHU TOAN BO job bi 429 va fail
hang loat (39/88 model-run fail trong lan chay dau). concurrency=4 CHAY
DON LE (khong song song) da chung minh an toan (job 1-3 khong 429 lan
nao). MAX_PARALLEL_JOBS=2 (toi da 8 dong thoi/model) la muc trung gian
duoc chon sau su co do - con day du margin ma van nhanh hon tuan tu ~2x.
Dung tang lai MAX_PARALLEL_JOBS ma khong kiem tra log rate-limit truoc.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv-kaggle/bin/python"
LOG_DIR = REPO / "results/frontier/openai/_logs/stage"

MAX_PARALLEL_JOBS = 2

JOBS = [
    ("ai_race/configs/experiment/openai_baseline.json", "results/frontier/openai/baseline"),
    ("ai_race/configs/experiment/openai_persona_baseline_neutral.json", "results/frontier/openai/persona/R0_neutral"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_averse.json", "results/frontier/openai/persona/Rminus_risk_averse"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_seeking.json", "results/frontier/openai/persona/Rplus_risk_seeking"),
    ("ai_race/configs/experiment/openai_persona_baseline_coop_coop.json", "results/frontier/openai/persona/S_CC_coop_coop"),
    ("ai_race/configs/experiment/openai_persona_baseline_adv_adv.json", "results/frontier/openai/persona/S_AA_adv_adv"),
    ("ai_race/configs/experiment/openai_persona_baseline_adv_coop.json", "results/frontier/openai/persona/S_AC_adv_coop"),
    ("ai_race/configs/experiment/openai_persona_baseline_coop_adv.json", "results/frontier/openai/persona/S_CA_coop_adv"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_1_1.json", "results/frontier/openai/persona/risk_matrix/R1_R1"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_1_2.json", "results/frontier/openai/persona/risk_matrix/R1_R2"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_1_3.json", "results/frontier/openai/persona/risk_matrix/R1_R3"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_1_4.json", "results/frontier/openai/persona/risk_matrix/R1_R4"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_1_5.json", "results/frontier/openai/persona/risk_matrix/R1_R5"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_1_6.json", "results/frontier/openai/persona/risk_matrix/R1_R6"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_2_1.json", "results/frontier/openai/persona/risk_matrix/R2_R1"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_2_2.json", "results/frontier/openai/persona/risk_matrix/R2_R2"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_2_3.json", "results/frontier/openai/persona/risk_matrix/R2_R3"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_2_4.json", "results/frontier/openai/persona/risk_matrix/R2_R4"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_2_5.json", "results/frontier/openai/persona/risk_matrix/R2_R5"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_2_6.json", "results/frontier/openai/persona/risk_matrix/R2_R6"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_3_1.json", "results/frontier/openai/persona/risk_matrix/R3_R1"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_3_2.json", "results/frontier/openai/persona/risk_matrix/R3_R2"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_3_3.json", "results/frontier/openai/persona/risk_matrix/R3_R3"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_3_4.json", "results/frontier/openai/persona/risk_matrix/R3_R4"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_3_5.json", "results/frontier/openai/persona/risk_matrix/R3_R5"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_3_6.json", "results/frontier/openai/persona/risk_matrix/R3_R6"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_4_1.json", "results/frontier/openai/persona/risk_matrix/R4_R1"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_4_2.json", "results/frontier/openai/persona/risk_matrix/R4_R2"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_4_3.json", "results/frontier/openai/persona/risk_matrix/R4_R3"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_4_4.json", "results/frontier/openai/persona/risk_matrix/R4_R4"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_4_5.json", "results/frontier/openai/persona/risk_matrix/R4_R5"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_4_6.json", "results/frontier/openai/persona/risk_matrix/R4_R6"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_5_1.json", "results/frontier/openai/persona/risk_matrix/R5_R1"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_5_2.json", "results/frontier/openai/persona/risk_matrix/R5_R2"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_5_3.json", "results/frontier/openai/persona/risk_matrix/R5_R3"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_5_4.json", "results/frontier/openai/persona/risk_matrix/R5_R4"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_5_5.json", "results/frontier/openai/persona/risk_matrix/R5_R5"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_5_6.json", "results/frontier/openai/persona/risk_matrix/R5_R6"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_6_1.json", "results/frontier/openai/persona/risk_matrix/R6_R1"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_6_2.json", "results/frontier/openai/persona/risk_matrix/R6_R2"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_6_3.json", "results/frontier/openai/persona/risk_matrix/R6_R3"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_6_4.json", "results/frontier/openai/persona/risk_matrix/R6_R4"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_6_5.json", "results/frontier/openai/persona/risk_matrix/R6_R5"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_6_6.json", "results/frontier/openai/persona/risk_matrix/R6_R6"),
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
