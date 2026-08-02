"""Data contracts and leakage-resistant splits for activation-level SAE work."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


ACTIVATION_SAE_SCHEMA_VERSION = "ai-race.activation-sae.v1"
_ACTION_RE = re.compile(r"\b(SAFE|UNSAFE)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class DecisionExample:
    """One enacted decision and the prefix immediately before its action token."""

    sample_id: str
    source_file: str
    source_row: int
    game_id: str
    player: str
    player_index: int
    round: int
    model: str
    prompt_version: str
    label_unsafe: int
    action: str
    prompt: str
    raw_response: str
    response_prefix: str
    prompt_sha256: str
    capture_prefix_sha256: str

    def metadata(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("prompt")
        row.pop("raw_response")
        row.pop("response_prefix")
        return row


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_action(value: Any) -> str:
    action = str(value or "").strip().lower()
    if action not in {"safe", "unsafe"}:
        raise ValueError(f"action must be SAFE or UNSAFE, got {value!r}")
    return action


def response_prefix_before_action(raw_response: str, action: str) -> str:
    """Return response text up to, but excluding, the enacted action label.

    The search is restricted to the recorded response, never the game prompt
    (which necessarily contains both action words).  The selected occurrence must
    match the parsed action, otherwise the row is rejected instead of silently
    introducing target leakage.
    """

    expected = _normalise_action(action)
    for match in _ACTION_RE.finditer(str(raw_response)):
        if match.group(1).lower() == expected:
            return str(raw_response)[: match.start(1)]
    raise ValueError(
        f"raw response does not contain parsed action {expected.upper()!r}"
    )


def build_capture_prefix(
    tokenizer: Any,
    prompt: str,
    response_prefix: str,
    *,
    position: str = "pre_action",
) -> str:
    """Render the causal prefix whose final token is captured.

    ``pre_action`` includes any chain-of-thought/reasoning and the common
    ``ACTION: `` prefix, but excludes SAFE/UNSAFE itself. ``prompt_last`` stops at
    the assistant generation marker and is useful as a stricter robustness view.
    """

    if position not in {"pre_action", "prompt_last"}:
        raise ValueError("position must be 'pre_action' or 'prompt_last'")
    messages = [{"role": "user", "content": str(prompt)}]
    template = getattr(tokenizer, "chat_template", None)
    if template:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        rendered = str(prompt)
    return rendered if position == "prompt_last" else rendered + str(response_prefix)


def _iter_turn_paths(inputs: Sequence[str | Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            paths.add(path.resolve())
        elif path.is_dir():
            paths.update(item.resolve() for item in path.rglob("turns.jsonl"))
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise FileNotFoundError("No turns.jsonl files found in --input-root values")
    return sorted(paths)


def load_decision_examples(
    inputs: Sequence[str | Path],
    *,
    exclude_parse_failures: bool = True,
) -> tuple[list[DecisionExample], dict[str, Any]]:
    """Load strict, auditable SAFE/UNSAFE rows from one or more result roots."""

    examples: list[DecisionExample] = []
    rejected: dict[str, int] = {}
    source_files: list[dict[str, Any]] = []
    for path in _iter_turn_paths(inputs):
        source_files.append(
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if exclude_parse_failures and bool(row.get("parse_failed", False)):
                        raise RuntimeError("parse_failed")
                    action = _normalise_action(row.get("action"))
                    label = int(action == "unsafe")
                    if row.get("unsafe") is not None and int(row["unsafe"]) != label:
                        raise RuntimeError("action_unsafe_conflict")
                    prompt = str(row["prompt"])
                    raw_response = str(row["raw_response"])
                    response_prefix = response_prefix_before_action(raw_response, action)
                    game_id = str(row["game_id"])
                    player = str(row.get("player", row.get("player_id", "")))
                    round_number = int(row["round"])
                    stable_key = "|".join(
                        (str(path), str(row_index), game_id, player, str(round_number))
                    )
                    examples.append(
                        DecisionExample(
                            sample_id=sha256_text(stable_key)[:24],
                            source_file=str(path),
                            source_row=row_index,
                            game_id=game_id,
                            player=player,
                            player_index=int(row.get("player_index", -1)),
                            round=round_number,
                            model=str(row.get("model", "")),
                            prompt_version=str(row.get("prompt_version", "")),
                            label_unsafe=label,
                            action=action,
                            prompt=prompt,
                            raw_response=raw_response,
                            response_prefix=response_prefix,
                            prompt_sha256=sha256_text(prompt),
                            capture_prefix_sha256=sha256_text(prompt + response_prefix),
                        )
                    )
                except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                    key = str(exc) or type(exc).__name__
                    rejected[key] = rejected.get(key, 0) + 1
    if not examples:
        raise ValueError("No valid, parsed SAFE/UNSAFE decisions were found")
    audit = {
        "schema_version": ACTIVATION_SAE_SCHEMA_VERSION,
        "n_examples": len(examples),
        "n_safe": sum(item.label_unsafe == 0 for item in examples),
        "n_unsafe": sum(item.label_unsafe == 1 for item in examples),
        "n_games": len({item.game_id for item in examples}),
        "models": sorted({item.model for item in examples}),
        "prompt_versions": sorted({item.prompt_version for item in examples}),
        "rejected": rejected,
        "source_files": source_files,
    }
    return examples, audit


def grouped_train_eval_split(
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    eval_fraction: float = 0.2,
    seed: int = 42,
    search_trials: int = 1024,
) -> list[str]:
    """Return train/eval assignments with no group crossing the boundary.

    A deterministic random search selects the group subset closest to both the
    requested row count and overall class prevalence. Missing a class in either
    split receives a prohibitive penalty. This is preferable to row-level random
    splitting because decisions inside one race are dependent.
    """

    if len(labels) != len(groups) or not labels:
        raise ValueError("labels and groups must be non-empty and equally sized")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between zero and one")
    label_values = [int(value) for value in labels]
    if any(value not in {0, 1} for value in label_values):
        raise ValueError("labels must be binary 0/1 values")
    unique_groups = sorted(set(str(group) for group in groups))
    if len(unique_groups) < 2:
        raise ValueError("at least two groups are required for grouped evaluation")

    by_group: dict[str, list[int]] = {group: [] for group in unique_groups}
    for label, group in zip(label_values, groups):
        by_group[str(group)].append(label)
    n_total = len(labels)
    target_n = n_total * eval_fraction
    target_rate = sum(label_values) / n_total
    rng = random.Random(seed)
    best_groups: set[str] | None = None
    best_score = float("inf")

    for _ in range(max(int(search_trials), 1)):
        order = unique_groups.copy()
        rng.shuffle(order)
        selected: list[str] = []
        selected_n = 0
        for group in order:
            if selected_n < target_n or not selected:
                selected.append(group)
                selected_n += len(by_group[group])
        eval_labels = [label for group in selected for label in by_group[group]]
        train_labels = [
            label for group in unique_groups if group not in selected for label in by_group[group]
        ]
        if not train_labels:
            continue
        eval_rate = sum(eval_labels) / len(eval_labels)
        missing_penalty = 0.0
        if len(set(eval_labels)) < min(2, len(set(label_values))):
            missing_penalty += 100.0
        if len(set(train_labels)) < min(2, len(set(label_values))):
            missing_penalty += 100.0
        score = (
            abs(len(eval_labels) - target_n) / n_total
            + abs(eval_rate - target_rate)
            + missing_penalty
        )
        if score < best_score:
            best_score = score
            best_groups = set(selected)
    if best_groups is None:
        raise ValueError("could not construct a non-empty grouped split")
    return ["eval" if str(group) in best_groups else "train" for group in groups]


def cross_split_duplicate_audit(
    hashes: Iterable[str], splits: Iterable[str]
) -> dict[str, Any]:
    locations: dict[str, set[str]] = {}
    for digest, split in zip(hashes, splits):
        locations.setdefault(str(digest), set()).add(str(split))
    crossed = sorted(digest for digest, values in locations.items() if len(values) > 1)
    return {
        "n_unique_prefixes": len(locations),
        "n_cross_split_duplicate_prefixes": len(crossed),
        "cross_split_duplicate_prefix_sha256": crossed,
    }


def race_prefix_components(
    race_ids: Sequence[str], prefix_hashes: Sequence[str]
) -> list[str]:
    """Build connected split groups that preserve races and exact-prefix duplicates.

    Rows are joined when they belong to the same race *or* share the same causal
    prefix. Taking the transitive closure prevents both within-race dependence
    and exact prompt leakage across train/eval.
    """

    if len(race_ids) != len(prefix_hashes) or not race_ids:
        raise ValueError("race_ids and prefix_hashes must be non-empty and equally sized")
    parent = list(range(len(race_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_race: dict[str, int] = {}
    first_prefix: dict[str, int] = {}
    for index, (race_id, prefix_hash) in enumerate(zip(race_ids, prefix_hashes)):
        race_key, prefix_key = str(race_id), str(prefix_hash)
        if race_key in first_race:
            union(index, first_race[race_key])
        else:
            first_race[race_key] = index
        if prefix_key in first_prefix:
            union(index, first_prefix[prefix_key])
        else:
            first_prefix[prefix_key] = index
    roots = [find(index) for index in range(len(race_ids))]
    canonical = {root: f"component-{rank:06d}" for rank, root in enumerate(sorted(set(roots)))}
    return [canonical[root] for root in roots]
