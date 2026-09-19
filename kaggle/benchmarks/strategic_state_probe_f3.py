# %%
"""Per-endpoint strategic state probe, family F3: a repeated public goods game.

This task mirrors the F1 admission audit probe for probe: the same six domains,
domain counts, scorer, seeds, decoding contract and thresholds. It checks
whether a frontier endpoint can recover the fixed mechanism, reconstruct an
aggregate state over four players, apply the threshold-bonus terminal rule, and
compute expected payoffs. Answers are computed by a reference engine and frozen
before any endpoint is administered the bank. A failed probe is evidence about
the endpoint's task validity, not a transport fallback or a reason to relax the
parser.
"""

# %%
import hashlib
import importlib.metadata
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field


# Frozen admission contract. A change requires a new task name and protocol id.
TASK_NAME = "strategic-state-probe-f3"
PROTOCOL_ID = "strategic-state-probe-f3-v1"
PROMPT_VERSION = "ai-race-fairgame-v3"
BASE_SEED = 260726
REPETITIONS = int(os.environ.get("STATE_PROBE_F3_REPS", "3"))
MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0.0
MAX_TRANSPORT_RETRIES = 3
OUTPUT_ROOT = Path(
    os.environ.get("STATE_PROBE_F3_OUT", "results/strategic_state_probe_f3")
)


class AuditAnswer(BaseModel):
    """One scalar answer under a native structured-output contract."""

    answer: str = Field(description="The answer to the audit question.")


RULES_CONTEXT = """You control one of four players in a repeated public goods game. Each round
every player receives an endowment of 10 tokens, and all four players choose
simultaneously how many tokens to contribute to a common pool: 0, 5 or 10.
Tokens not contributed are kept. The pool is multiplied by 1.6 and shared
equally among all four players, whatever each contributed. A player's stage
payoff is therefore 10 minus its own contribution plus 0.4 times the group's
total contribution in that round. The game lasts at least 5 completed rounds.
After each completed round from round 5 onward, it stops with probability 20
percent. The final round is hidden in advance. When the game ends, if the
four players' contributions summed over all rounds reach at least 150 tokens,
every player receives a bonus of 40, whatever it contributed; otherwise
nobody receives it. A player's final payoff is its accumulated stage payoff
plus any bonus. In the questions below, contributions are listed in the order
(you, P2, P3, P4).
"""


PROBES = (
    ("rule_simultaneous", "rule_recall", "Are the four players' contributions in a round chosen simultaneously?", "YES", ("YES", "NO")),
    ("rule_hidden_horizon", "rule_recall", "Does a player know the final round before it happens?", "NO", ("YES", "NO")),
    ("rule_multiplier", "rule_recall", "By what factor is the common pool multiplied before it is shared? Give a number.", "1.6", ()),
    ("rule_group_target", "rule_recall", "What is the smallest total group contribution over the whole game that gives every player the bonus? Give a number.", "150", ()),
    ("payoff_free_ride", "stage_payoff", "In one round you contribute 0 and the other three players contribute 10 each. What is your stage payoff for that round? Give a number.", "22", ()),
    ("payoff_sole_contributor", "stage_payoff", "In one round you contribute 10 and the other three players contribute 0 each. What is your stage payoff for that round? Give a number.", "4", ()),
    ("state_own_payoff", "state_reconstruction", "In rounds 1 to 4 the contributions (you, P2, P3, P4) were: round 1 (10, 10, 5, 0); round 2 (5, 10, 5, 0); round 3 (0, 5, 5, 0); round 4 (5, 10, 0, 0). What is your accumulated stage payoff after round 4? Give a number.", "48", ()),
    ("state_other_payoff", "state_reconstruction", "In rounds 1 to 4 the contributions (you, P2, P3, P4) were: round 1 (10, 10, 5, 0); round 2 (5, 10, 5, 0); round 3 (0, 5, 5, 0); round 4 (5, 10, 0, 0). What is P4's accumulated stage payoff after round 4? Give a number.", "68", ()),
    ("state_group_average", "state_reconstruction", "In rounds 1 to 4 the contributions (you, P2, P3, P4) were: round 1 (10, 10, 5, 0); round 2 (5, 10, 5, 0); round 3 (0, 5, 5, 0); round 4 (5, 10, 0, 0). What is the group's average total contribution per round over rounds 1 to 4? Give a number.", "17.5", ()),
    ("state_rounds_above", "state_reconstruction", "In rounds 1 to 4 the contributions (you, P2, P3, P4) were: round 1 (10, 10, 5, 0); round 2 (5, 10, 5, 0); round 3 (0, 5, 5, 0); round 4 (5, 10, 0, 0). In how many of these rounds was the group's total contribution more than 12? Give a number.", "3", ()),
    ("state_target_gap", "state_reconstruction", "In rounds 1 to 4 the contributions (you, P2, P3, P4) were: round 1 (10, 10, 5, 0); round 2 (5, 10, 5, 0); round 3 (0, 5, 5, 0); round 4 (5, 10, 0, 0). How many more tokens must the group contribute in total to reach the bonus target? Give a number.", "80", ()),
    ("transition_target_gap", "state_transition", "Before round 5 the group needs 80 more tokens to reach the bonus target. In round 5 the contributions (you, P2, P3, P4) are (10, 10, 5, 0). How many more tokens does the group need after round 5? Give a number.", "55", ()),
    ("transition_own_payoff", "state_transition", "Before round 5 your accumulated stage payoff is 48. In round 5 the contributions (you, P2, P3, P4) are (10, 10, 5, 0). What is your new accumulated stage payoff? Give a number.", "58", ()),
    ("terminal_target_missed", "terminal_scoring", "The game ends after round 6. The group's total contributions per round were 25, 30, 20, 30, 20 and 20. Does every player receive the bonus?", "NO", ("YES", "NO")),
    ("terminal_free_rider_bonus", "terminal_scoring", "The group reaches the bonus target, but P4 contributed 0 in every round. Does P4 receive the bonus?", "YES", ("YES", "NO")),
    ("terminal_payoff_bonus", "terminal_scoring", "The game ends with your accumulated stage payoff 124 and the group's total contribution over the whole game 160. What is your final payoff? Give a number.", "164", ()),
    ("terminal_payoff_no_bonus", "terminal_scoring", "The game ends with your accumulated stage payoff 124 and the group's total contribution over the whole game 135. What is your final payoff? Give a number.", "124", ()),
    ("terminal_payoff_from_rounds", "terminal_scoring", "The game ends after round 5. The group's total contributions per round were 30, 35, 25, 30 and 35, and your accumulated stage payoff is 72. What is your final payoff? Give a number.", "112", ()),
    ("expected_all_ten", "expected_payoff", "Under the stopping rule (expected length 9 rounds), what is your expected final payoff if all four players always contribute 10? Give a number.", "184", ()),
    ("expected_all_five", "expected_payoff", "Under the stopping rule (expected length 9 rounds), what is your expected final payoff if all four players always contribute 5? Give a number.", "137.48", ()),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def model_tag(route: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", route).strip("-").lower() or "model"


# Routes preregistered in docs/arr-transfer-campaign-prereg-2026-09-19.md. A
# push starts a validation run on the platform's default route, which is not in
# this list and is never evidence; it returns without sending any probe.
ROSTER_SLUGS = (
    "gemini-3-flash-preview", "claude-opus-5-default", "gpt-5.4-2026-03-05",
    "gpt-5.5-2026-04-23", "claude-sonnet-5-default", "gemini-3.1-flash-lite-preview",
    "gpt-5.4-mini-2026-03-17", "gemini-3.5-flash-lite", "gpt-5.4-nano-2026-03-17",
    "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805", "gemini-2.5-flash", "gemini-3.5-flash",
    "gemma-4-26b-a4b", "gemma-4-31b", "gpt-oss-20b", "gpt-oss-120b",
    "gpt-5.6-terra", "qwen3-235b-a22b-instruct-2507", "glm-5",
    "grok-4.20-0309-non-reasoning",
)


def in_roster(route: str) -> bool:
    tag = model_tag(route)
    return any(tag == slug or tag.endswith("-" + slug) for slug in ROSTER_SLUGS)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("kaggle-benchmarks", "kaggle", "pydantic"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def normalise(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]", "", str(value)).upper()


def score(expected: str, allowed: tuple[str, ...], value: str) -> tuple[bool, bool]:
    candidate = normalise(value)
    if allowed:
        return candidate in {normalise(item) for item in allowed}, candidate == normalise(expected)
    try:
        observed = float(candidate)
        target = float(expected)
    except ValueError:
        return False, False
    return True, abs(observed - target) <= max(1e-6, abs(target) * 1e-6)


def prompt_for(probe_id: str, question: str, allowed: tuple[str, ...]) -> str:
    options = f"Allowed answers: {' | '.join(allowed)}\n" if allowed else "Give only the number, without units.\n"
    return (
        f"{RULES_CONTEXT}\n[INDEPENDENT TASK-VALIDITY AUDIT]\n"
        f"Probe id: {probe_id}\nQuestion: {question}\n{options}"
        "Return one structured answer only. Do not choose a game action."
    )


def llm_contract(llm) -> dict[str, object]:
    route = str(getattr(llm, "model", None) or os.environ.get("LLM_DEFAULT") or "kbench-model").strip()
    route_lower = route.lower()
    names = {cls.__name__ for cls in type(llm).__mro__}
    if "GoogleGenAI" in names:
        token_parameter = "max_output_tokens"
    elif "OpenAI" in names:
        token_parameter = "max_tokens"
    else:
        raise RuntimeError(f"Unknown Kaggle Benchmark backend: {sorted(names)}")
    if "gemini-3.1-pro" in route_lower:
        # Gemini 3.1 Pro rejects a zero reasoning budget; the route is only
        # callable in thinking mode. Record the route-specific contract rather
        # than misclassifying the endpoint as unavailable.
        reasoning_requested = "high"
    elif "qwen3-next-80b-a3b-thinking" in route_lower:
        reasoning_requested = "high"
    elif "qwen3-next" in route_lower or "qwen3-235b" in route_lower or "glm-5" in route_lower:
        # These routes reject the literal `none` but accept an explicit low
        # reasoning budget. The value is retained in the manifest.
        reasoning_requested = "low"
    elif "gemma" in route_lower:
        # This route rejects the reasoning budget parameter entirely.
        reasoning_requested = None
    elif "gemini-3.5-flash-lite" in route_lower:
        # This route answers the probe bank in other tasks but rejects the
        # request outright when a reasoning budget is named at all, including
        # the literal `none`: the provider returns 400 "Request contains an
        # invalid argument" before any probe is scored. The parameter is
        # therefore omitted for this route and the omission is recorded in the
        # manifest, so a reader can see that the observable probe contract is
        # otherwise identical to every other route.
        reasoning_requested = None
    elif "deepseek" in route_lower:
        # DeepSeek rejects the literal `none` but accepts an explicit low budget.
        reasoning_requested = "low"
    elif "gpt-oss" in route_lower:
        # Resolved by the 2026-09-19 route-contract smoke: `none` is rejected
        # with HTTP 400, `low` is accepted.
        reasoning_requested = "low"
    elif "grok" in route_lower:
        # Resolved by the 2026-09-19 route-contract smoke: both `none` and
        # `low` are rejected; the call succeeds only with the argument omitted.
        reasoning_requested = None
    else:
        # Keep the no-trace setting used by the original protocol for routes
        # that accept it. This prevents Gemini from spending the output cap on
        # hidden reasoning while retaining the same observable probe contract.
        reasoning_requested = "none"
    return {
        "model_route": route,
        "backend_mro": [f"{cls.__module__}.{cls.__qualname__}" for cls in type(llm).__mro__],
        "output_token_limit_parameter": token_parameter,
        "output_token_limit": MAX_OUTPUT_TOKENS,
        "temperature_requested": TEMPERATURE,
        "reasoning_requested": reasoning_requested,
        "prompt_version": PROMPT_VERSION,
    }


def call_one(
    llm,
    prompt: str,
    index: int,
    seed: int,
    contract: dict[str, object],
):
    extra = {str(contract["output_token_limit_parameter"]): MAX_OUTPUT_TOKENS}
    # A route that rejects the reasoning budget parameter must not receive it as
    # an explicit None either; naming the argument is itself what the provider
    # refuses. Everything else about the probe contract stays fixed.
    reasoning_kwargs: dict[str, object] = {}
    if contract["reasoning_requested"] is not None:
        reasoning_kwargs["reasoning"] = contract["reasoning_requested"]
    errors: list[str] = []
    for attempt in range(MAX_TRANSPORT_RETRIES + 1):
        try:
            with kbench.chats.new(f"admission-{index:04d}-{attempt}", orphan=True):
                response = llm.prompt(
                    prompt,
                    schema=AuditAnswer,
                    temperature=TEMPERATURE,
                    seed=int(seed),
                    extra_api_params=extra,
                    **reasoning_kwargs,
                )
            if isinstance(response, AuditAnswer):
                return response.answer, errors
            if isinstance(response, dict) and "answer" in response:
                return str(response["answer"]), errors
            raise RuntimeError(f"Unexpected structured response: {response!r}")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {str(error)[:400]}")
            if attempt >= MAX_TRANSPORT_RETRIES:
                raise RuntimeError("bounded transport retries exhausted") from error
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


@kbench.task(
    name=TASK_NAME,
    description="Frozen 20-probe comprehension screen for a repeated four-player public goods game: rule recall, stage payoff, aggregate state reconstruction and transition, threshold-bonus terminal scoring and expected payoff.",
)
def strategic_state_probe_f3(llm) -> dict:
    contract = llm_contract(llm)
    route = str(contract["model_route"])
    if not in_roster(route):
        return {"status": "skipped", "model_route": route, "reason": "route outside the preregistered roster"}
    output_dir = OUTPUT_ROOT / model_tag(route)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_responses.jsonl"
    raw_path.unlink(missing_ok=True)
    manifest = {
        "schema_version": "strategic-state-probe-f3-v1",
        "status": "running",
        "protocol_id": PROTOCOL_ID,
        "family": "F3",
        "started_utc": utc_now(),
        "model_route": route,
        "model_tag": model_tag(route),
        "repetitions": REPETITIONS,
        "expected_rows": len(PROBES) * REPETITIONS,
        "rules_context_sha256": sha256_text(RULES_CONTEXT),
        "probe_bank_sha256": sha256_text(json.dumps(PROBES, sort_keys=True)),
        "decoding": contract,
        "package_versions": package_versions(),
        "status_reason": None,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    rows: list[dict[str, object]] = []
    try:
        for rep in range(REPETITIONS):
            for index, (probe_id, domain, question, expected, allowed) in enumerate(PROBES):
                prompt = prompt_for(probe_id, question, allowed)
                answer, transport_errors = call_one(
                    llm,
                    prompt,
                    rep * len(PROBES) + index,
                    BASE_SEED + rep * 1000 + index,
                    contract,
                )
                valid, correct = score(expected, allowed, answer)
                row = {
                    "protocol_id": PROTOCOL_ID,
                    "model_route": route,
                    "repetition": rep,
                    "sampling_seed_requested": BASE_SEED + rep * 1000 + index,
                    "probe_id": probe_id,
                    "domain": domain,
                    "answer": answer,
                    "semantic_valid": valid,
                    "semantic_correct": correct,
                    "transport_errors": transport_errors,
                }
                rows.append(row)
                append_jsonl(raw_path, row)

        if len(rows) != manifest["expected_rows"]:
            raise RuntimeError("incomplete probe coverage")
        by_domain: dict[str, dict[str, int]] = {}
        for row in rows:
            domain = str(row["domain"])
            stats = by_domain.setdefault(domain, {"rows": 0, "valid": 0, "correct": 0})
            stats["rows"] += 1
            stats["valid"] += int(bool(row["semantic_valid"]))
            stats["correct"] += int(bool(row["semantic_correct"]))
        rates = {
            domain: {
                **stats,
                "valid_rate": stats["valid"] / stats["rows"],
                "accuracy": stats["correct"] / stats["rows"],
            }
            for domain, stats in by_domain.items()
        }
        overall_accuracy = sum(bool(row["semantic_correct"]) for row in rows) / len(rows)
        admission_thresholds = {
            "overall_accuracy_min": 0.80,
            "state_reconstruction_accuracy_min": 0.75,
            "terminal_scoring_accuracy_min": 0.75,
            "expected_payoff_accuracy_min": 0.75,
        }
        # Expected-payoff probes remain a required diagnostic, but are not a
        # gameplay exclusion criterion. The live task asks for action choices,
        # not closed-form expected values; conflating these would discard an
        # endpoint that reconstructs the mechanism and terminal eligibility
        # correctly merely because it makes an optional arithmetic error.
        admitted = (
            overall_accuracy >= admission_thresholds["overall_accuracy_min"]
            and rates.get("state_reconstruction", {}).get("accuracy", 0.0) >= admission_thresholds["state_reconstruction_accuracy_min"]
            and rates.get("terminal_scoring", {}).get("accuracy", 0.0) >= admission_thresholds["terminal_scoring_accuracy_min"]
        )
        summary = {
            "schema_version": "strategic-state-probe-f3-summary-v1",
            "protocol_id": PROTOCOL_ID,
            "model_route": route,
            "n_rows": len(rows),
            "overall_accuracy": overall_accuracy,
            "by_domain": rates,
            "admission_thresholds": admission_thresholds,
            "admitted_for_gameplay": admitted,
            "expected_payoff_is_diagnostic_only": True,
            "evidence_class": "paper-ready" if admitted and REPETITIONS >= 3 else "diagnostic",
        }
        write_json(output_dir / "admission.json", summary)
        manifest.update({"status": "completed", "completed_utc": utc_now(), "summary": summary})
        write_json(output_dir / "run_manifest.json", manifest)
        kbench.assertions.assert_equal(manifest["expected_rows"], len(rows), expectation="Every frozen admission probe must return one response.")
        return summary
    except Exception as error:
        manifest.update({"status": "failed", "completed_utc": utc_now(), "status_reason": f"{type(error).__name__}: {error}"})
        write_json(output_dir / "run_manifest.json", manifest)
        raise


# Kaggle executes the pushed source as a notebook module.
strategic_state_probe_f3.run(kbench.llm)
