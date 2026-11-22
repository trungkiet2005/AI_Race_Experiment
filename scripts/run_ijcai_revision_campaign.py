"""Run the slices of docs/ijcai-revision-collection-plan-2026-11-23.md.

Each slice runs on its declared identity through run_ijcai_revision_slice.sh.
Amendment 1 of the plan is applied mechanically: an identity refused with a
quota 403 during push validation, before any assigned route started, is
replaced by the next unused identity in the declared reserve order, and the
refusal is written as a failure record. Any other failure is left as it is.
Threads only wait on subprocesses; nothing heavy runs on this machine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_ijcai_revision_slice.sh"
FAILED = ROOT / "results" / "failed_runs"
LEDGER = ROOT / "results" / "_build" / "ijcai_revision" / "campaign_ledger.jsonl"

SLICES = [
    ("boymagic", "AS", "0.1", "B", "admitted"),
    ("chisboiz", "AS", "0.6", "B", "admitted"),
    ("chunaiu", "AS", "0.9", "B", "admitted"),
    ("daosyduyminh", "AU", "0.1", "B", "admitted"),
    ("hnhhctiliu", "AU", "0.6", "B", "admitted"),
    ("kakagotto", "AU", "0.9", "B", "admitted"),
    ("minh2duy", "AS", "0.1", "C", "admitted"),
    ("osduyminh", "AS", "0.6", "C", "admitted"),
    ("tonngohan", "AS", "0.9", "C", "admitted"),
    ("trngbotrn", "AU", "0.1", "C", "admitted"),
    ("vinhdinhthien", "AU", "0.6", "C", "admitted"),
    ("trunkdabest", "AU", "0.9", "C", "admitted"),
    ("chiboiz", "AS", "0.1", "A", "refused"),  # reserve 1, replaces kit567 (amendment 1)
    ("tnkiet", "AS", "0.6", "A", "refused"),
    ("hunhtrungkit", "AS", "0.9", "A", "refused"),
    ("foundnotkiet", "AU", "0.1", "A", "refused"),
    ("trungkiet", "AU", "0.6", "A", "refused"),
    ("trngthtnhi", "AU", "0.9", "A", "refused"),
]
RESERVE = ["chinguyentran", "trnnguynchis", "vkhoinguyen", "nguyendinhhaduong", "nvviet05", "tntduong"]
_lock = threading.Lock()


def log(event: dict) -> None:
    event["utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with _lock:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)


def next_reserve() -> str | None:
    with _lock:
        return RESERVE.pop(0) if RESERVE else None


def run_slice(ident: str, strat: str, risk: str, block: str, group: str) -> None:
    while ident:
        log({"event": "start", "identity": ident, "slice": [block, strat, risk, group]})
        proc = subprocess.run(
            ["bash", str(DRIVER), ident, strat, risk, block, group],
            capture_output=True, text=True,
        )
        out = proc.stdout + proc.stderr
        quota_at_push = (
            proc.returncode == 4
            and "exceeds your available quota" in out
            and "== run " not in out
        )
        log({"event": "end", "identity": ident, "slice": [block, strat, risk, group],
             "returncode": proc.returncode,
             "ingested": out.count("ingested "), "refused": out.count("INGEST_REFUSED"),
             "quota_at_push": quota_at_push})
        if not quota_at_push:
            return
        name = f"ijcai_revision_{block}_{strat}_{risk.replace('.', 'p')}_{ident}_push403.json"
        FAILED.mkdir(parents=True, exist_ok=True)
        (FAILED / name).write_text(json.dumps({
            "schema": "failure-record-v1",
            "plan": "docs/ijcai-revision-collection-plan-2026-11-23.md",
            "slice": {"block": block, "strategy": strat, "risk": float(risk), "group": group},
            "identity": ident,
            "stage": "push validation on gemini-3-flash-preview (not an assigned route)",
            "error": "HTTP 403 quota refusal",
            "assigned_routes_started": 0,
            "classification": "quota refusal; never model evidence; replaced under amendment 1",
        }, indent=2), encoding="utf-8")
        ident = next_reserve()
        if ident is None:
            log({"event": "no_reserve_left", "slice": [block, strat, risk, group]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="comma list of identities to run")
    args = ap.parse_args()
    todo = [s for s in SLICES if not args.only or s[0] in args.only.split(",")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for s in todo:
            pool.submit(run_slice, *s)


if __name__ == "__main__":
    main()
