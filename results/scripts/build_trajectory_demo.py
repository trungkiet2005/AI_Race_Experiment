#!/usr/bin/env python3
"""Package the audited trajectory payload for the standalone demo."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "impact_upgrade" / "data" / "trajectory_demo.json"
TARGET = ROOT / "docs" / "demos" / "trajectory_lab" / "data.js"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ai-race-trajectory-demo-v1":
        raise ValueError("Unexpected trajectory demo schema")
    if not payload.get("cases"):
        raise ValueError("Trajectory demo has no cases")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(
        "window.AI_RACE_TRAJECTORY_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "target": str(TARGET), "cases": len(payload["cases"])}, indent=2))


if __name__ == "__main__":
    main()
