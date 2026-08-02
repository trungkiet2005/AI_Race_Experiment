from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_race.xai.activation_sae import (
    build_capture_prefix,
    cross_split_duplicate_audit,
    grouped_train_eval_split,
    load_decision_examples,
    race_prefix_components,
    response_prefix_before_action,
)


class _Tokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>"


def _write_turns(path: Path) -> None:
    rows = [
        {
            "game_id": "g1",
            "player": "P1",
            "player_index": 0,
            "round": 1,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt_version": "v3",
            "action": "unsafe",
            "unsafe": 1,
            "parse_failed": False,
            "prompt": "Rules mention SAFE and UNSAFE.",
            "raw_response": "Reasoning. ACTION: UNSAFE",
        },
        {
            "game_id": "g2",
            "player": "P2",
            "player_index": 1,
            "round": 1,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt_version": "v3",
            "action": "safe",
            "unsafe": 0,
            "parse_failed": False,
            "prompt": "Rules mention SAFE and UNSAFE.",
            "raw_response": "ACTION: SAFE",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_response_prefix_excludes_decision_label():
    prefix = response_prefix_before_action("analysis\nACTION: UNSAFE", "unsafe")
    assert prefix == "analysis\nACTION: "
    assert "UNSAFE" not in prefix


def test_response_prefix_rejects_conflicting_label():
    with pytest.raises(ValueError, match="does not contain"):
        response_prefix_before_action("ACTION: SAFE", "unsafe")


def test_build_capture_prefix_uses_chat_template_without_target():
    rendered = build_capture_prefix(
        _Tokenizer(), "rules", "ACTION: ", position="pre_action"
    )
    assert rendered == "<user>rules</user><assistant>ACTION: "


def test_loader_validates_action_and_builds_provenance(tmp_path: Path):
    turns = tmp_path / "turns.jsonl"
    _write_turns(turns)
    examples, audit = load_decision_examples([tmp_path])
    assert [item.label_unsafe for item in examples] == [1, 0]
    assert examples[0].response_prefix.endswith("ACTION: ")
    assert audit["n_games"] == 2
    assert audit["source_files"][0]["sha256"]


def test_group_split_never_splits_a_race_and_keeps_classes():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    splits = grouped_train_eval_split(labels, groups, eval_fraction=0.5, seed=7)
    for group in set(groups):
        assert len({split for split, item_group in zip(splits, groups) if item_group == group}) == 1
    for split in ("train", "eval"):
        assert {label for label, item_split in zip(labels, splits) if item_split == split} == {0, 1}


def test_duplicate_audit_reports_cross_split_prefixes():
    audit = cross_split_duplicate_audit(["x", "x", "y"], ["train", "eval", "train"])
    assert audit["n_cross_split_duplicate_prefixes"] == 1
    assert audit["cross_split_duplicate_prefix_sha256"] == ["x"]


def test_race_prefix_components_close_both_dependency_edges():
    groups = race_prefix_components(
        ["race-a", "race-a", "race-b", "race-c"],
        ["p1", "p2", "p2", "p3"],
    )
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
