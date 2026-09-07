# %%
"""Hosted, paper-scale N=3 AI-race baseline for Kaggle Benchmarks.

The benchmark deliberately imports the tested N-player engine from the attached
``nguyenlamphuquy/ai-race-experiment`` dataset instead of duplicating its payoff,
prompt, RNG, and recording logic.  Each model decision is requested through a
fresh orphan chat with a native two-value schema.  That keeps simultaneous moves
independent and avoids treating provider prose-format differences as game actions.

The source defaults are the prospectively frozen run: three risk treatments,
60 common-environment-seed repetitions per treatment, neutral symmetric N=3
self-play, and ``run_phase=confirmatory``.  Environment overrides exist only for
local smoke diagnostics; the hosted revision used for evidence records every
effective value in ``run_manifest.json``.
"""

# %%
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field, ValidationError


# %%
TASK_NAME = "ai-race-nplayer-baseline-n3"
PROTOCOL_ID = "ai-race-nplayer-n3-hosted-confirmatory-v2"
N_PLAYERS = 3
RISK_LEVELS = (0.1, 0.6, 0.9)
CONFIRMATORY_REPETITIONS = 60
CREATION_VALIDATION_REPETITIONS = 1
BASE_SEED = int(os.environ.get("AI_RACE_SEED", "260802"))
REPETITIONS_OVERRIDE = os.environ.get("AI_RACE_REPS")
RUN_PHASE_OVERRIDE = os.environ.get("AI_RACE_RUN_PHASE")

CONFIRMATORY_MODEL_ROUTES = (
    "openai/gpt-5.4-nano-2026-03-17",
    "google/gemini-3.5-flash-lite",
    "anthropic/claude-haiku-4-5@20251001",
)

TEMPERATURE = 0.7
REASONING = "none"
MAX_OUTPUT_TOKENS = 64
MAX_PARSE_RETRIES = 3
MAX_TRANSPORT_RETRIES = 8
REQUEST_TIMEOUT_SECONDS = 120

REPO_INPUT_DIRS = (
    Path("/kaggle/input/datasets/nguyenlamphuquy/ai-race-experiment"),
    Path("/kaggle/input/ai-race-experiment"),
)


class ActionDecision(BaseModel):
    """Native structured-output contract for one game decision."""

    action: Literal["SAFE", "UNSAFE"] = Field(
        description="The company's action for this simultaneous round."
    )


# %%
def _is_repo_root(path: Path) -> bool:
    return (path / "ai_race" / "engine_nplayer").is_dir() and (
        path / "ai_race" / "configs"
    ).is_dir()


def _find_repo_root() -> Path:
    candidates: list[Path] = []
    try:
        source = Path(__file__).resolve()
        candidates.extend(source.parents)
    except NameError:
        pass
    candidates.extend([Path.cwd(), *Path.cwd().parents, *REPO_INPUT_DIRS])
    for candidate in candidates:
        if _is_repo_root(candidate):
            return candidate.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for engine_dir in kaggle_input.glob("*/*/ai_race/engine_nplayer"):
            candidate = engine_dir.parents[1]
            if _is_repo_root(candidate):
                return candidate.resolve()
        for engine_dir in kaggle_input.glob("*/ai_race/engine_nplayer"):
            candidate = engine_dir.parents[1]
            if _is_repo_root(candidate):
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the attached AI Race repository. Push this task with "
        "-d nguyenlamphuquy/ai-race-experiment."
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_race.dataio.config_loader import load_json  # noqa: E402
from ai_race.engine_nplayer.recorder import (  # noqa: E402
    NPlayerRunJournal,
    write_players_csv,
    write_races_csv,
)
from ai_race.engine_nplayer.runner import (  # noqa: E402
    build_games_for_model,
    run_games_batched,
)

CONFIG_DIR = REPO_ROOT / "ai_race" / "configs"
EXPERIMENT_PATH = CONFIG_DIR / "experiment" / "baseline_nplayer_n3.json"
AGENTS_PATH = (
    CONFIG_DIR / "agents_nplayer" / "companies_nplayer_default_n3.json"
)
GAME_PATHS = tuple(
    CONFIG_DIR / "game_nplayer" / f"ai_race_nplayer_n3_risk_{label}.json"
    for label in ("10", "60", "90")
)
PROMPT_PATH = (
    REPO_ROOT
    / "ai_race"
    / "engine_nplayer"
    / "prompts"
    / "ai_race_nplayer_en.txt"
)


# %%
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_tag(route: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", route).strip("-").lower()


def _is_confirmatory_model(route: str) -> bool:
    return route.strip().lower() in {
        item.lower() for item in CONFIRMATORY_MODEL_ROUTES
    }


def _effective_run_settings(route: str) -> dict:
    target_model = _is_confirmatory_model(route)
    repetitions = (
        int(REPETITIONS_OVERRIDE)
        if REPETITIONS_OVERRIDE is not None
        else (
            CONFIRMATORY_REPETITIONS
            if target_model
            else CREATION_VALIDATION_REPETITIONS
        )
    )
    run_phase = (
        RUN_PHASE_OVERRIDE.strip().lower()
        if RUN_PHASE_OVERRIDE is not None
        else ("confirmatory" if target_model else "pilot")
    )
    if repetitions < 1:
        raise ValueError("AI_RACE_REPS must be positive")
    if run_phase not in {"pilot", "confirmatory"}:
        raise ValueError("AI_RACE_RUN_PHASE must be pilot or confirmatory")
    if run_phase == "confirmatory" and (
        not target_model or repetitions != CONFIRMATORY_REPETITIONS
    ):
        raise RuntimeError(
            "Confirmatory execution is restricted to the three frozen model "
            f"routes at exactly {CONFIRMATORY_REPETITIONS} repetitions."
        )
    return {
        "target_model": target_model,
        "repetitions": repetitions,
        "run_phase": run_phase,
        "execution_profile": (
            "confirmatory" if run_phase == "confirmatory" else "creation_validation"
        ),
    }


def _output_dir(route: str) -> Path:
    override = os.environ.get("AI_RACE_BENCHMARK_OUT")
    if override:
        return Path(override)
    return Path("results/ai_race_nplayer_baseline_n3") / _model_tag(route)


def _task_source_runtime_sha256() -> str:
    """Hash the executed notebook cell file; the local source hash is external.

    Kaggle converts the pushed percent-script into notebook cells, so ``__file__``
    is not byte-identical to the local source.  The orchestrator separately pins
    the local file SHA-256 and remote task revision.
    """
    try:
        source = Path(__file__)
        if source.is_file():
            return _sha256_file(source)
    except NameError:
        pass
    frozen = repr(
        (
            TASK_NAME,
            PROTOCOL_ID,
            N_PLAYERS,
            RISK_LEVELS,
            CONFIRMATORY_REPETITIONS,
            BASE_SEED,
            CONFIRMATORY_MODEL_ROUTES,
            TEMPERATURE,
            REASONING,
            MAX_OUTPUT_TOKENS,
        )
    )
    return hashlib.sha256(frozen.encode("utf-8")).hexdigest()


def _engine_source_sha256() -> str:
    digest = hashlib.sha256()
    roots = (
        REPO_ROOT / "ai_race" / "engine",
        REPO_ROOT / "ai_race" / "engine_nplayer",
        REPO_ROOT / "ai_race" / "dataio",
    )
    files = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".txt"}
    )
    for path in files:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _input_file_hashes() -> dict[str, str]:
    paths = (EXPERIMENT_PATH, AGENTS_PATH, *GAME_PATHS, PROMPT_PATH)
    return {
        path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path)
        for path in paths
    }


def _mechanism_snapshot() -> dict:
    games = [load_json(path) for path in GAME_PATHS]
    shared_fields = (
        "engine",
        "nPlayers",
        "safeProgress",
        "speed",
        "cost",
        "benefit",
        "minRounds",
        "stopProbability",
        "maxRoundsSafetyCap",
        "racePrize",
        "historyMode",
        "promptTemplate",
        "promptVersion",
        "agents",
    )
    snapshot = {field: games[0][field] for field in shared_fields}
    for game in games[1:]:
        for field, expected in snapshot.items():
            if game[field] != expected:
                raise RuntimeError(
                    f"Game configs disagree on mechanism field {field!r}"
                )
    snapshot["maxPrivateRiskTreatments"] = [
        game["maxPrivateRisk"] for game in games
    ]
    return snapshot


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("kaggle-benchmarks", "kaggle", "openai", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _reset_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "turns.jsonl",
        "races.csv",
        "players.csv",
        "summary.json",
        "run_manifest.json",
        "request_errors.jsonl",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _load_experiment(
    model_route: str,
    *,
    repetitions: int,
    run_phase: str,
) -> dict:
    experiment = load_json(EXPERIMENT_PATH)
    experiment.update(
        {
            "name": "hosted_nplayer_n3_confirmatory_baseline",
            "runPhase": run_phase,
            "description": (
                "Prospectively frozen hosted N=3 neutral baseline: three risk "
                "treatments with matched environment seeds."
            ),
            "models": [model_route],
            "useOffline": False,
            "backend": "kaggle_benchmarks",
            "samplingSeedApplied": False,
            "repetitions": repetitions,
            "seed": BASE_SEED,
            "maxParseRetries": MAX_PARSE_RETRIES,
            "verbose": False,
        }
    )
    return experiment


def _counterbalanced_games(games: list) -> list:
    """Order games by repetition, cyclically rotating risk within each block."""
    by_rep: dict[int, dict[float, object]] = defaultdict(dict)
    for game in games:
        risk = float(game.config.max_private_risk)
        if risk in by_rep[int(game.rep)]:
            raise RuntimeError(f"Duplicate risk {risk} in repetition {game.rep}")
        by_rep[int(game.rep)][risk] = game

    ordered = []
    risks = list(RISK_LEVELS)
    for rep in sorted(by_rep):
        if set(by_rep[rep]) != set(risks):
            raise RuntimeError(
                f"Repetition {rep} does not contain all frozen risk treatments"
            )
        shift = rep % len(risks)
        for risk in risks[shift:] + risks[:shift]:
            ordered.append(by_rep[rep][risk])
    return ordered


# %%
def _llm_contract(llm) -> dict:
    route = str(
        getattr(llm, "model", None)
        or os.environ.get("LLM_DEFAULT")
        or "kbench-model"
    ).strip()
    backend_mro = [
        f"{cls.__module__}.{cls.__qualname__}" for cls in type(llm).__mro__
    ]
    backend_names = {cls.__name__ for cls in type(llm).__mro__}
    if "GoogleGenAI" in backend_names:
        token_parameter = "max_output_tokens"
        timeout_applied = False
    elif "OpenAI" in backend_names:
        token_parameter = "max_tokens"
        timeout_applied = True
    else:
        raise RuntimeError(
            "Unknown Kaggle Benchmark LLM backend; refusing to omit the output "
            f"cap. backend_mro={backend_mro}"
        )

    seed_strip_probe = getattr(llm, "_should_remove_seed", None)
    seed_forwarded = None
    if callable(seed_strip_probe):
        try:
            seed_forwarded = not bool(seed_strip_probe())
        except Exception:
            seed_forwarded = None

    return {
        "model_route": route,
        "backend_mro": backend_mro,
        "output_token_limit_parameter": token_parameter,
        "output_token_limit": MAX_OUTPUT_TOKENS,
        "request_timeout_applied": timeout_applied,
        "temperature_requested": TEMPERATURE,
        "temperature_forwarded_by_sdk": bool(
            getattr(llm, "support_temperature", False)
        ),
        "reasoning_requested": REASONING,
        "sampling_seed_requested": True,
        "sampling_seed_forwarded_by_sdk": seed_forwarded,
        "sampling_seed_applied": None,
        "sampling_seed_applied_known": False,
    }


class ProtocolResponseError(RuntimeError):
    """A model response violated the frozen structured-output contract."""


def _transport_status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        value = getattr(getattr(error, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_transport_error(error: Exception) -> bool:
    if _transport_status_code(error) in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(error).__name__.lower()
    return any(
        marker in name
        for marker in (
            "timeout",
            "connectionerror",
            "connecterror",
            "networkerror",
            "remotedisconnected",
        )
    )


class StructuredKBenchBackend:
    """Adapt one KBench LLM actor to ``send_batch(prompts, seeds)``."""

    def __init__(self, llm, contract: dict, error_log_path: Path):
        self.llm = llm
        self.contract = contract
        self.error_log_path = error_log_path
        self.decision_attempt_count = 0
        self.request_count = 0
        self.transport_retry_count = 0
        self.request_error_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_cost_nanodollars = 0
        self.output_cost_nanodollars = 0

    @staticmethod
    def _usage_value(usage, name: str) -> int:
        value = getattr(usage, name, 0) if usage is not None else 0
        return int(value or 0)

    def _one(self, prompt: str, seed: int, request_index: int) -> dict:
        errors: list[dict] = []
        for transport_attempt in range(MAX_TRANSPORT_RETRIES + 1):
            chat_name = (
                f"n3-{request_index:06d}-transport{transport_attempt + 1}"
            )
            try:
                with kbench.chats.new(chat_name, orphan=True) as chat:
                    extra_api_params = {
                        self.contract["output_token_limit_parameter"]: (
                            self.contract["output_token_limit"]
                        )
                    }
                    if self.contract["request_timeout_applied"]:
                        extra_api_params["timeout"] = REQUEST_TIMEOUT_SECONDS
                    response = self.llm.prompt(
                        prompt,
                        schema=ActionDecision,
                        reasoning=REASONING,
                        temperature=TEMPERATURE,
                        seed=int(seed),
                        extra_api_params=extra_api_params,
                    )
                    usage = getattr(chat, "usage", None)

                self.request_count += 1
                self.input_tokens += self._usage_value(usage, "input_tokens")
                self.output_tokens += self._usage_value(usage, "output_tokens")
                self.input_cost_nanodollars += self._usage_value(
                    usage, "input_tokens_cost_nanodollars"
                )
                self.output_cost_nanodollars += self._usage_value(
                    usage, "output_tokens_cost_nanodollars"
                )
                raw = (
                    response.model_dump_json()
                    if isinstance(response, BaseModel)
                    else json.dumps(response, ensure_ascii=False)
                    if isinstance(response, dict)
                    else str(response)
                )
                action = (
                    response.action
                    if isinstance(response, ActionDecision)
                    else response.get("action")
                    if isinstance(response, dict)
                    else None
                )
                if action not in {"SAFE", "UNSAFE"}:
                    raise ProtocolResponseError(
                        f"Structured response has invalid action: {raw!r}"
                    )
                return {
                    "text": f"ACTION: {action}",
                    "raw_response": raw,
                    "transport_errors": errors,
                }
            except (ProtocolResponseError, ValidationError) as error:
                event = {
                    "request_index": request_index,
                    "transport_attempt": transport_attempt,
                    "classification": "protocol_response_failure",
                    "retryable": False,
                    "status_code": _transport_status_code(error),
                    "error_type": type(error).__name__,
                    "error": str(error)[:4000],
                }
                self.request_error_count += 1
                _append_jsonl(self.error_log_path, event)
                raise ProtocolResponseError(
                    "Protocol failure: native structured response did not "
                    "satisfy ActionDecision; no fallback action was applied."
                ) from error
            except Exception as error:
                retryable = _is_retryable_transport_error(error)
                event = {
                    "request_index": request_index,
                    "transport_attempt": transport_attempt,
                    "classification": (
                        "retryable_transport" if retryable else "nonretryable_request"
                    ),
                    "retryable": retryable,
                    "status_code": _transport_status_code(error),
                    "error_type": type(error).__name__,
                    "error": str(error)[:4000],
                }
                errors.append(event)
                self.request_error_count += 1
                _append_jsonl(self.error_log_path, event)
                if not retryable:
                    raise RuntimeError(
                        "Non-retryable model request failure; no fallback game "
                        "action was applied."
                    ) from error
                if transport_attempt >= MAX_TRANSPORT_RETRIES:
                    raise RuntimeError(
                        "Kaggle Model Proxy request failed after bounded retries; "
                        "no fallback game action was applied."
                    ) from error
                self.transport_retry_count += 1
                time.sleep(min(2**transport_attempt, 30))
        raise AssertionError("unreachable")

    def __call__(self, prompts, seeds=None):
        prompts = list(prompts)
        seed_values = list(seeds) if seeds is not None else [0] * len(prompts)
        if len(seed_values) != len(prompts):
            raise ValueError("seeds must align with prompts")
        responses = []
        for prompt, seed in zip(prompts, seed_values):
            self.decision_attempt_count += 1
            responses.append(
                self._one(prompt, int(seed), self.decision_attempt_count)
            )
        return responses

    def usage_summary(self) -> dict:
        return {
            "decision_attempts": self.decision_attempt_count,
            "requests": self.request_count,
            "transport_retries": self.transport_retry_count,
            "request_errors": self.request_error_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost_nanodollars": self.input_cost_nanodollars,
            "output_cost_nanodollars": self.output_cost_nanodollars,
            "total_cost_usd": (
                self.input_cost_nanodollars + self.output_cost_nanodollars
            )
            / 1_000_000_000,
        }


# %%
def _summarize(
    results,
    journal: NPlayerRunJournal,
    backend,
    *,
    model_route: str,
    repetitions: int,
    output_dir: Path,
    execution_profile: str,
) -> dict:
    risk_counts = {
        str(risk): sum(
            1 for result in results if result.max_private_risk == risk
        )
        for risk in RISK_LEVELS
    }
    parse_failures = sum(result.parse_failures for result in results)
    mean_rounds = (
        sum(result.n_rounds for result in results) / len(results)
        if results
        else None
    )
    return {
        "task_name": TASK_NAME,
        "protocol_id": PROTOCOL_ID,
        "execution_profile": execution_profile,
        "model": _model_tag(model_route),
        "model_route": model_route,
        "n_players": N_PLAYERS,
        "n_races": len(results),
        "n_player_rows": len(results) * N_PLAYERS,
        "n_decisions": journal.turn_count,
        "repetitions_per_risk": repetitions,
        "risk_counts": risk_counts,
        "parse_failures": parse_failures,
        "mean_rounds": mean_rounds,
        "usage": backend.usage_summary(),
        "output_dir": str(output_dir),
    }


@kbench.task(
    name=TASK_NAME,
    description=(
        "Three-player repeated AI race: neutral symmetric self-play across "
        "private setback risks 0.1, 0.6, and 0.9 with 60 matched-seed races "
        "per risk treatment."
    ),
)
def ai_race_nplayer_baseline_n3(llm) -> dict:
    contract = _llm_contract(llm)
    settings = _effective_run_settings(contract["model_route"])
    repetitions = settings["repetitions"]
    run_phase = settings["run_phase"]
    output_dir = _output_dir(contract["model_route"])
    _reset_outputs(output_dir)
    experiment = _load_experiment(
        contract["model_route"],
        repetitions=repetitions,
        run_phase=run_phase,
    )
    manifest = {
        "schema_version": "ai-race-nplayer-kbench-run-v2",
        "status": "running",
        "protocol_id": PROTOCOL_ID,
        "execution_profile": settings["execution_profile"],
        "run_phase": run_phase,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": None,
        "task_name": TASK_NAME,
        "model": _model_tag(contract["model_route"]),
        "model_route": contract["model_route"],
        "n_players": N_PLAYERS,
        "task_source_runtime_sha256": _task_source_runtime_sha256(),
        "engine_source_sha256": _engine_source_sha256(),
        "prompt_sha256": _sha256_file(PROMPT_PATH),
        "agents_config_sha256": _sha256_file(AGENTS_PATH),
        "input_file_sha256": _input_file_hashes(),
        "mechanism": _mechanism_snapshot(),
        "experiment": experiment,
        "request_order": {
            "race_order": "repetition-blocked cyclic risk rotation",
            "risk_rotation": "rep r starts at risk index r mod 3",
            "seat_order_within_race": [0, 1, 2],
            "backend_execution": "sequential",
        },
        "decoding": {
            **contract,
            "structured_output": True,
            "response_schema": ActionDecision.model_json_schema(),
            "max_parse_retries": MAX_PARSE_RETRIES,
            "max_transport_retries": MAX_TRANSPORT_RETRIES,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "native_structured_response_retained_in_turns": False,
            "native_response_limitation": (
                "The attached N-player engine stores the canonical ACTION line "
                "in turns.jsonl; schema validity is enforced before conversion."
            ),
        },
        "package_versions": _package_versions(),
        "expected_races": len(RISK_LEVELS) * repetitions,
        "n_races": 0,
        "n_player_rows": 0,
        "n_turns": 0,
        "usage": None,
        "error": None,
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    journal = NPlayerRunJournal(output_dir, reset=False)
    backend = StructuredKBenchBackend(
        llm,
        contract,
        output_dir / "request_errors.jsonl",
    )
    results = []
    try:
        games = build_games_for_model(experiment, contract["model_route"])
        if len(games) != manifest["expected_races"]:
            raise RuntimeError(
                f"Expected {manifest['expected_races']} games, built {len(games)}"
            )
        games = _counterbalanced_games(games)
        results = run_games_batched(
            games,
            backend,
            verbose=False,
            max_parse_retries=MAX_PARSE_RETRIES,
            on_round_complete=journal.record_round,
        )
        summary = _summarize(
            results,
            journal,
            backend,
            model_route=contract["model_route"],
            repetitions=repetitions,
            output_dir=output_dir,
            execution_profile=settings["execution_profile"],
        )
        expected_risk_counts = {str(risk): repetitions for risk in RISK_LEVELS}
        if summary["n_races"] != manifest["expected_races"]:
            raise RuntimeError("Incomplete race coverage")
        if summary["n_player_rows"] != manifest["expected_races"] * N_PLAYERS:
            raise RuntimeError("Incomplete player coverage")
        if summary["risk_counts"] != expected_risk_counts:
            raise RuntimeError(
                f"Risk coverage mismatch: {summary['risk_counts']}"
            )
        if summary["parse_failures"] != 0:
            raise RuntimeError(
                "Protocol failure: at least one model decision failed parsing"
            )

        write_races_csv(results, output_dir / "races.csv")
        write_players_csv(results, output_dir / "players.csv")
        _write_json(output_dir / "summary.json", summary)
        manifest.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "n_races": summary["n_races"],
                "n_player_rows": summary["n_player_rows"],
                "n_turns": summary["n_decisions"],
                "usage": summary["usage"],
            }
        )
        _write_json(output_dir / "run_manifest.json", manifest)

        kbench.assertions.assert_equal(
            manifest["expected_races"],
            summary["n_races"],
            expectation="All preregistered N=3 races must complete.",
        )
        kbench.assertions.assert_equal(
            0,
            summary["parse_failures"],
            expectation="Every decision must satisfy ActionDecision.",
        )
        return summary
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "n_races": journal.race_count,
                "n_player_rows": journal.player_count,
                "n_turns": journal.turn_count,
                "usage": backend.usage_summary(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(output_dir / "run_manifest.json", manifest)
        raise


# %%
# Kaggle executes the pushed source as a notebook module.
ai_race_nplayer_baseline_n3.run(kbench.llm)
