"""Consolidate every AI Race figure (52 already-admitted + 33 new exploratory)
into one browsable folder with a single master index, for paper figure
selection. Does not modify results/open_source/ or the gallery folder.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


def _long(p: Path) -> str:
    """Windows MAX_PATH (260 chars) workaround for the deeply nested scratchpad path."""
    s = str(p.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s

REPO_SRC = Path(r"d:\AI_PhD\GameTheory\AI_Race_Experiment\results\open_source")
GALLERY = Path(r"C:\Users\huynh\AppData\Local\Temp\claude\d--AI-PhD-GameTheory-AI-Race-Experiment\352ed5ca-cf7a-45f3-a367-16150ebaa721\scratchpad\ai_race_gallery")
OUT = Path(r"C:\Users\huynh\AppData\Local\Temp\claude\d--AI-PhD-GameTheory-AI-Race-Experiment\352ed5ca-cf7a-45f3-a367-16150ebaa721\scratchpad\ai_race_all_figures")

entries = []  # dict(section, section_title, name, title, caption, status)


def add(section, section_title, src: Path, name: str, title: str, caption: str, status: str):
    dest_dir = OUT / section
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.copy2(_long(src), _long(dest))
    entries.append({
        "section": section, "section_title": section_title,
        "relpath": f"{section}/{name}", "title": title, "caption": caption, "status": status,
    })


# ---------------------------------------------------------------------------
# 1. new exploratory figures -- parse the gallery INDEX.md we already wrote
# ---------------------------------------------------------------------------
NEW_SECTION_MAP = {
    "00_overview": "0. Cross-pilot overview  [NEW]",
    "01_persona": "1. Persona-sensitivity pilot (prompt_sensitivity_pilot/)  [NEW]",
    "02_surface": "2. Surface-sensitivity pilot + smoke  [NEW]",
    "03_game_understanding": "3. Game-understanding pilot  [NEW]",
    "04_context_supplement": "4. Context-skin pilot -- new supplementary views  [NEW]",
}

index_text = (GALLERY / "INDEX.md").read_text(encoding="utf-8")
# split on "### " entries, capturing title / image relpath / caption
pattern = re.compile(
    r"### (?P<title>.+?)\n\n!\[.*?\]\((?P<relpath>.+?)\)\n\n(?P<caption>.+?)\n\n", re.S)
for m in pattern.finditer(index_text + "\n\n"):
    relpath = m.group("relpath")
    section_key = relpath.split("/")[0]
    section_title = NEW_SECTION_MAP.get(section_key, section_key)
    out_section = f"NEW_{section_key}"
    src = GALLERY / "figures" / relpath
    name = Path(relpath).name
    add(out_section, section_title, src, name, m.group("title"), m.group("caption").strip(), "new")

print("copied new figures:", sum(1 for e in entries))

# ---------------------------------------------------------------------------
# 2. already-admitted figures already in the repo
# ---------------------------------------------------------------------------
ADMITTED_CAPTION = {
    ("context_skin_pilot", "analysis_live_pilot_t0"):
        "Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.",
    ("context_skin_pilot", "analysis_live_pilot_t07"):
        "T=0.7 decoding-sensitivity companion run, same design as the T=0 primary pilot.",
    ("context_skin_pilot", "analysis_smoke_t0"):
        "Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.",
    ("context_skin_pilot", "analysis_temperature_robustness"):
        "Paired T=0 vs T=0.7 robustness analysis, 10,000 race-cluster bootstrap replicates.",
    ("context_skin_pilot", "context_recognition_t0_pilot"):
        "Frozen context-recognition audit (does the model name the real-world domain it's role-playing?).",
    ("egt_reproduction", None):
        "Reduced evolutionary-game-theory reconstruction of the source paper, and its LLM comparison.",
    ("activation_sae", "causal_selfplay/fast-sae-pilot-L12-v1/analysis"):
        "Native-runtime causal self-play SAE audit (dose/sign/reconstruction/matched-random controls, layer 12).",
    ("activation_sae", "context_fast_sae_analysis"):
        "Layer-12/layer-20 context screen: held-out probe AUC and causal-steering controls.",
    ("activation_sae", "figures"):
        "Primary token-position comparison across the pre-action vs prompt-last capture positions.",
    ("activation_sae", "surface_n600_strict_pre_action/figures"):
        "SAE fidelity/probe/feature-confirmation figures, pre-action capture position (n=600 decisions).",
    ("activation_sae", "surface_n600_strict_prompt_last/figures"):
        "Same views at the stricter prompt-last capture position, before response boilerplate.",
    ("prompt_sensitivity_pilot", "xai_auto_vector_encoder"):
        "Surrogate logged-feature XAI model (global coefficients / permutation importance), full prompt features.",
    ("prompt_sensitivity_pilot", "xai_auto_vector_encoder_no_response"):
        "Same surrogate XAI model with response-derived features removed.",
}

ADMITTED_SECTION_TITLE = {
    "context_skin_pilot": "5. Context-skin pilot -- admitted figures  [ADMITTED]",
    "egt_reproduction": "6. EGT reconstruction -- admitted figures  [ADMITTED]",
    "activation_sae": "7. Activation-level SAE pilot -- admitted figures  [ADMITTED]",
    "prompt_sensitivity_pilot": "8. Persona pilot -- surrogate XAI figures  [ADMITTED]",
}

for src in sorted(REPO_SRC.rglob("*.png")):
    rel = src.relative_to(REPO_SRC)
    parts = rel.parts
    pilot = parts[0]
    subdir = "/".join(parts[1:-1]) if len(parts) > 2 else None
    section = f"ADMITTED_{pilot}"
    section_title = ADMITTED_SECTION_TITLE.get(pilot, pilot)
    caption = ADMITTED_CAPTION.get((pilot, subdir))
    if caption is None:
        # fall back to nearest known prefix match (activation_sae has nested variants)
        for (p, s), c in ADMITTED_CAPTION.items():
            if p == pilot and subdir and s and subdir.startswith(s):
                caption = c
                break
    if caption is None:
        caption = f"From {rel.parent.as_posix()}/."
    tag = subdir.replace("/", "__") if subdir else pilot
    name = f"{tag}__{src.name}" if subdir else src.name
    title = src.stem.replace("_", " ")
    add(section, section_title, src, name, title, caption, "admitted")

print("total copied:", len(entries))

# ---------------------------------------------------------------------------
# write master index
# ---------------------------------------------------------------------------
section_order = list(dict.fromkeys(e["section"] for e in entries))
# order: new sections in original order, then admitted sections in original order
new_order = [f"NEW_{k}" for k in NEW_SECTION_MAP]
admitted_order = [f"ADMITTED_{k}" for k in ["context_skin_pilot", "egt_reproduction", "activation_sae", "prompt_sensitivity_pilot"]]
final_order = [s for s in new_order + admitted_order if s in section_order]

lines = [
    "# AI Race -- all figures, gathered for paper selection",
    "",
    f"**{len(entries)} figures total**: 33 new exploratory (never published anywhere) + "
    f"{len(entries) - 33} already-admitted figures copied in from `results/open_source/*`. "
    "`[NEW]` sections are descriptive-only, generated outside the repo's admission process. "
    "`[ADMITTED]` sections passed the project's own hash/manifest admission gates -- see each pilot's "
    "README under `results/open_source/` for the exact evidence-class wording before using one in the paper.",
    "",
    "Folder layout mirrors the sections below (`NEW_01_persona/`, `ADMITTED_context_skin_pilot/`, ...); "
    "every file in this folder is a plain copy, safe to delete or move independently.",
    "",
]
for sec in final_order:
    rows = [e for e in entries if e["section"] == sec]
    if not rows:
        continue
    lines.append(f"## {rows[0]['section_title']}")
    lines.append("")
    for r in rows:
        lines.append(f"### {r['title']}")
        lines.append("")
        lines.append(f"![{r['title']}]({r['relpath']})")
        lines.append("")
        lines.append(r["caption"])
        lines.append("")

(OUT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT / "INDEX.md")

by_status = {}
for e in entries:
    by_status[e["status"]] = by_status.get(e["status"], 0) + 1
print("by status:", by_status)
