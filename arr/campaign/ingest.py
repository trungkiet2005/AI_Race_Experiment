"""Ingest downloaded ARR transfer-campaign runs into the results tree.

Reads a download root laid out as <identity>/<task>/<version>/<model>/<run_id>/
(the layout `kaggle b t download` writes), and copies each run into
results/frontier/arr_transfer_campaign/<family>/<model>/<run_id>/ when it
completed with a full probe set, or into .../failed_runs/<family>/<model>/<run_id>/
with its failure class otherwise. Push-validation runs on routes outside the
preregistered roster are skipped. Identities are written as neutral labels; the
label-to-account map stays outside the repository.

    python arr/campaign/ingest.py D:/kaggle/working/dl/arr
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results" / "frontier" / "arr_transfer_campaign"
FAMILY_BY_TASK = {
    "ai-race-frontier-admission": "F1",
    "strategic-state-probe-f2": "F2",
    "strategic-state-probe-f3": "F3",
}
# The label-to-account map lives outside this public repository, beside the
# credential inventory; override its location with ARR_LABEL_MAP.
LABEL_MAP = Path(os.environ.get(
    "ARR_LABEL_MAP",
    r"D:\PhD_LetGoo\PhD_Farming\infra\kaggle_for_research\ai_race_arr_label_map.json",
))
LABEL_BY_IDENTITY = json.loads(LABEL_MAP.read_text(encoding="utf-8")) if LABEL_MAP.is_file() else {}
VALIDATION_ONLY = {"gemini-3.7-flash"}
EXPECTED_ROWS = 60


def failure_class(message: str) -> str:
    if "403" in message and "quota" in message.lower():
        return "quota_refusal"
    if "LengthFinishReasonError" in message:
        return "reasoning_cap_exhausted"
    if "ResponseParsingError" in message or "ValidationError" in message:
        return "structured_output_not_honoured"
    if "Error code: 503" in message or "Error code: 429" in message or "Error code: 502" in message:
        return "transport"
    return "other"


def raw_rows(run_dir: Path) -> int:
    files = list(run_dir.glob("results/*/*/raw_responses.jsonl"))
    if len(files) != 1:
        return -1
    return sum(1 for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    source = Path(sys.argv[1])
    ledger = []
    for run_json in sorted(source.glob("*/*/*/*/*/*.run.json")):
        run_dir = run_json.parent
        run_id = run_dir.name
        model = run_dir.parent.name
        version = run_dir.parent.parent.name
        task = run_dir.parent.parent.parent.name
        identity = run_dir.parent.parent.parent.parent.name
        if model in VALIDATION_ONLY:
            continue
        family = FAMILY_BY_TASK[task]
        record = json.loads(run_json.read_text(encoding="utf-8"))
        state = str(record.get("state", ""))
        rows = raw_rows(run_dir)
        completed = state.endswith("COMPLETED") and rows == EXPECTED_ROWS
        cls = "" if completed else failure_class(str(record.get("errorMessage") or ""))
        if state.endswith("COMPLETED") and rows != EXPECTED_ROWS:
            cls = f"completed_with_{rows}_rows"
        target = DEST / family / model / run_id if completed else DEST / "failed_runs" / family / model / run_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run_dir, target)
        ledger.append({
            "family": family, "model": model, "task_version": version, "run_id": run_id,
            "collector": LABEL_BY_IDENTITY.get(identity, "unlabelled"),
            "state": state.replace("BENCHMARK_TASK_RUN_STATE_", "").lower(),
            "raw_rows": rows, "admitted_to_analysis": completed, "failure_class": cls,
        })

    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in ledger:
        by_cell.setdefault((row["family"], row["model"]), []).append(row)
    for (family, model), rows in by_cell.items():
        done = [r for r in rows if r["admitted_to_analysis"]]
        if len(done) > 1:
            raise SystemExit(f"{family}/{model}: {len(done)} completed runs; a cell is collected once")

    DEST.mkdir(parents=True, exist_ok=True)
    fields = list(ledger[0].keys())
    with (DEST / "ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(ledger, key=lambda r: (r["family"], r["model"], r["run_id"])))
    ok = sum(r["admitted_to_analysis"] for r in ledger)
    print(f"{len(ledger)} runs ingested, {ok} complete cells, {len(ledger) - ok} failure records")


if __name__ == "__main__":
    main()
