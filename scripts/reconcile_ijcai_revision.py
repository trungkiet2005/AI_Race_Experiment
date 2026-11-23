"""Bring every declared slice to completion from its real server-side state.

For each (identity, slice) in the final assignment, read the benchmark task
status on that identity and pick the resume mode of run_ijcai_revision_slice.sh:
  done     every assigned route is already ingested locally
  collect  the missing routes have runs on the server (finished or running)
  run      the pushed version is validated or validating, no route run yet
  push     nothing usable on the server
The driver is copied to a temporary file first, so editing the repository copy
can never change a slice that is already running.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = Path.home() / "PhD_Farming/infra/kaggle_for_research/scripts/kaggle_pool.py"
TASK = "ai-race-scripted-opponent"
ADMITTED = ["gemini-3-flash-preview", "gpt-5.4-2026-03-05", "gpt-5.5-2026-04-23",
            "claude-sonnet-5-default", "claude-opus-5-default"]
REFUSED = ["gemini-3.1-flash-lite-preview", "gpt-5.4-mini-2026-03-17",
           "gemini-3.5-flash-lite", "gpt-5.4-nano-2026-03-17"]

# Final assignment after amendments 1 to 4 and the replacement log.
SLICES = [
    ("vkhoinguyen", "AS", "0.1", "B", "admitted"),
    ("chisboiz", "AS", "0.6", "B", "admitted"),
    ("chunaiu", "AS", "0.9", "B", "admitted"),
    ("nguyendinhhaduong", "AU", "0.1", "B", "admitted"),
    ("hnhhctiliu", "AU", "0.6", "B", "admitted"),
    ("kakagotto", "AU", "0.9", "B", "admitted"),
    ("minh2duy", "AS", "0.1", "C", "admitted"),
    ("tntduong", "AS", "0.6", "C", "admitted"),
    ("tonngohan", "AS", "0.9", "C", "admitted"),
    ("trngbotrn", "AU", "0.1", "C", "admitted"),
    ("vinhdinhthien", "AU", "0.6", "C", "admitted"),
    ("trunkdabest", "AU", "0.9", "C", "admitted"),
    ("nvviet05", "AS", "0.1", "A", "refused"),
    ("hunhtrungkit", "AS", "0.9", "A", "refused"),
    ("foundnotkiet", "AU", "0.1", "A", "refused"),
    ("trungkiet", "AU", "0.6", "A", "refused"),
    ("trngthtnhi", "AU", "0.9", "A", "refused"),
]
FAILURE_RECORDED = {("chisboiz", "gpt-5.5-2026-04-23")}  # quota 403 after the route started


def campaign(block: str, group: str) -> Path:
    if group == "refused":
        return ROOT / "results/frontier/scripted_opponent_refused"
    return ROOT / f"results/frontier/scripted_opponent_replication/block_{block}"


def ingested(model, strat, risk, block, group) -> bool:
    cell = campaign(block, group) / f"{strat}_risk{risk.replace('.', 'p')}" / model
    return (cell / "collection_receipt.json").exists()


def status(ident: str) -> tuple[str, dict]:
    out = subprocess.run(["python3", str(POOL), "run", ident, "--", "b", "t", "status", TASK],
                         capture_output=True, text=True).stdout
    task_state, models = "", {}
    for line in out.splitlines():
        if line.startswith("Status:"):
            task_state = line.split()[1]
        parts = line.split()
        if len(parts) >= 2 and (parts[0] in ADMITTED or parts[0] in REFUSED):
            models[parts[0]] = parts[1]
    return task_state, models


def decide(s):
    ident, strat, risk, block, group = s
    models = ADMITTED if group == "admitted" else REFUSED
    missing = [m for m in models
               if not ingested(m, strat, risk, block, group) and (ident, m) not in FAILURE_RECORDED]
    if not missing:
        return "done", "", {}, missing
    task_state, on_server = status(ident)
    if set(on_server) & set(missing):
        return "collect", task_state, on_server, missing
    if task_state in ("Running", "Completed"):
        return "run", task_state, on_server, missing
    return "push", task_state, on_server, missing


def run(s, mode: str, driver: Path) -> None:
    env = dict(os.environ)
    if mode in ("run", "collect"):
        env["SKIP_PUSH"] = "1"
    if mode == "collect":
        env["SKIP_RUN"] = "1"
    log = ROOT / "results/_build/ijcai_revision" / f"reconcile_{s[0]}.out"
    with log.open("a") as fh:
        proc = subprocess.run(["bash", str(driver), *s], stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(json.dumps({"event": "reconciled", "identity": s[0], "mode": mode,
                      "returncode": proc.returncode}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--modes", default="collect,run,push",
                    help="which resume modes to execute; others are reported only")
    args = ap.parse_args()
    plan = []
    for s in SLICES:
        mode, state, on_server, missing = decide(s)
        plan.append((s, mode))
        print(json.dumps({"identity": s[0], "slice": list(s[1:]), "mode": mode, "task": state,
                          "server_routes": on_server, "missing": missing}), flush=True)
    if args.dry_run:
        return
    tmp = Path(tempfile.mkdtemp(prefix="ijcai_driver_"))
    driver = tmp / "run_ijcai_revision_slice.sh"
    text = (ROOT / "scripts/run_ijcai_revision_slice.sh").read_text()
    text = text.replace('REPO="$(cd "$(dirname "$0")/.." && pwd)"', f'REPO="{ROOT}"')
    driver.write_text(text)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for s, mode in plan:
            if mode in args.modes.split(","):
                pool.submit(run, s, mode, driver)


if __name__ == "__main__":
    main()
