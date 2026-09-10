"""Move one downloaded matched-sweep cell into the repository, with its receipt.

The sweep is collected one cell at a time on whichever Kaggle identity has
quota, so ingestion happens repeatedly and by hand is where a provenance mistake
would creep in. This does it the same way every time.

Two things it records that the run itself cannot. The executing identity, because
the benchmark server exposes neither ``KAGGLE_USERNAME`` nor
``KAGGLE_KERNEL_RUN_OWNER`` to task code, so the run manifest's own field reads
``unrecorded`` and the account has to be supplied by whoever ran it. And the
declaring document, so a cell can always be traced back to the plan that fixed
its assignment before it was collected.

It refuses to ingest a run that did not complete, that carries a parse failure,
that is not the frozen protocol, or that would silently overwrite a different
cell already in the tree.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "frontier" / "nplayer_matched_campaign"
PROTOCOL_ID = "ai-race-nplayer-matched-hosted-confirmatory-v1"
DEFAULT_PLAN = "docs/matched-nplayer-collection-plan-2026-09-10.md"


def find_run(download_root: Path) -> tuple[Path, dict]:
    """Locate the one completed sweep run under a download root."""
    candidates: list[tuple[Path, dict]] = []
    for path in sorted(download_root.rglob("run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "by_group_size" in manifest:
            candidates.append((path.parent, manifest))
    if not candidates:
        raise SystemExit(f"no matched-sweep run found under {download_root}")
    completed = [(p, m) for p, m in candidates if m.get("status") == "completed"]
    if not completed:
        statuses = ", ".join(sorted({str(m.get("status")) for _, m in candidates}))
        raise SystemExit(
            f"no completed run under {download_root}; statuses seen: {statuses}. "
            "A refused or interrupted run is a failure record, not a cell."
        )
    if len(completed) > 1:
        raise SystemExit(
            f"{len(completed)} completed runs under {download_root}; clear the "
            "download root between cells so the wrong one cannot be ingested"
        )
    return completed[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument(
        "--identity",
        required=True,
        help="Kaggle account that ran the cell; the server cannot tell us.",
    )
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a cell already in the tree. Off by default so a repeat "
        "download cannot quietly overwrite a different collection.",
    )
    args = parser.parse_args()

    src, manifest = find_run(args.download_root)

    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit(
            f"protocol is {manifest.get('protocol_id')!r}, not {PROTOCOL_ID}"
        )
    sizes = manifest.get("group_sizes") or []
    risks = manifest.get("risk_levels") or []
    if len(sizes) != 1 or len(risks) != 1:
        raise SystemExit(
            f"expected one cell, found group sizes {sizes} and risks {risks}"
        )
    n_players, risk = int(sizes[0]), float(risks[0])
    summary = manifest["by_group_size"][str(n_players)]
    if summary["parse_failures"]:
        raise SystemExit(
            f"{summary['parse_failures']} parse failures: one failure "
            "contaminates its race, so this cell is a failure record"
        )

    dst = CAMPAIGN / f"n{n_players}_risk{str(risk).replace('.', 'p')}" / "gemini-3-flash-preview"
    if dst.exists() and not args.force:
        existing = dst / "collection_receipt.json"
        who = (
            json.loads(existing.read_text(encoding="utf-8")).get("executing_identity")
            if existing.exists()
            else "unknown"
        )
        raise SystemExit(
            f"{dst.relative_to(ROOT)} already holds a cell collected on {who}; "
            "pass --force only if you mean to replace it"
        )
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)

    receipt = {
        "schema_version": "matched-cell-collection-receipt-v1",
        "cell": {"n_players": n_players, "max_private_risk": risk},
        "executing_identity": args.identity,
        "identity_source": (
            "Supplied at ingestion, because the benchmark server exposes no "
            "identity variable to task code; the run manifest's own field "
            "therefore reads 'unrecorded'."
        ),
        "model_route": "google/gemini-3-flash-preview",
        "declared_in": args.plan,
        "protocol_id": PROTOCOL_ID,
        "n_races": summary["n_races"],
        "n_decisions": summary["n_decisions"],
        "parse_failures": summary["parse_failures"],
    }
    (dst / "collection_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )

    print(
        f"ingested N={n_players} risk={risk} on {args.identity}: "
        f"{summary['n_races']} races, {summary['n_decisions']} decisions, "
        f"{summary['parse_failures']} parse failures"
    )
    print(f"  -> {dst.relative_to(ROOT)}")

    have = sorted(
        (
            int(p.parent.parent.name.split("_")[0][1:]),
            p.parent.parent.name.split("risk")[1].replace("p", "."),
        )
        for p in CAMPAIGN.glob("*/*/collection_receipt.json")
    )
    print(f"  campaign now holds {len(have)} cells: {have}")


if __name__ == "__main__":
    main()
