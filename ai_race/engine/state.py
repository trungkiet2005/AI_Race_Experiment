"""Typed configuration and output records for the AI Race environment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Action(str, Enum):
    """The two development actions used by the experiment."""

    SAFE = "safe"
    UNSAFE = "unsafe"

    @property
    def label(self) -> str:
        return self.value.upper()

    @property
    def unsafe(self) -> int:
        return int(self is Action.UNSAFE)

    @classmethod
    def coerce(cls, value: "Action | str") -> "Action":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {"s": cls.SAFE, "safe": cls.SAFE, "u": cls.UNSAFE, "unsafe": cls.UNSAFE}
        if normalized not in aliases:
            raise ValueError(f"Unknown AI Race action: {value!r}")
        return aliases[normalized]


@dataclass
class GameConfig:
    """Game mechanism loaded from ``configs/game/*.json``.

    JSON files use camelCase to stay readable from notebook configuration cells.
    Internal Python attributes use snake_case.
    """

    name: str
    engine: str = "ai_race"
    n_players: int = 2
    safe_progress: float = 1.0
    unsafe_progress: float = 1.5
    payoff_safe_safe: float = 1.0
    payoff_safe_unsafe: float = 0.6
    payoff_unsafe_safe: float = 2.4
    payoff_unsafe_unsafe: float = 2.0
    min_rounds: int = 5
    stop_probability: float = 0.2
    max_rounds_safety_cap: int = 100
    race_prize: float = 100.0
    max_private_risk: float = 0.1
    history_mode: str = "previous_round"
    agents_communicate: bool = False
    reveal_opponent_persona_prior: bool = False
    language: str = "en"
    prompt_template: str = "ai_race_en"
    prompt_version: str = "ai-race-paper-v2"
    run_phase: str = "pilot"
    agents_ref: str = "companies_default"
    model: str = "LocalQwen"
    sampling_seed_applied: bool = True

    def __post_init__(self) -> None:
        if self.engine != "ai_race":
            raise ValueError(f"Unsupported engine {self.engine!r}; expected 'ai_race'")
        if self.n_players != 2:
            raise ValueError("The paper-faithful AI Race requires exactly two players")
        if self.safe_progress <= 0 or self.unsafe_progress <= self.safe_progress:
            raise ValueError("Unsafe progress must be greater than positive Safe progress")
        if self.min_rounds < 1:
            raise ValueError("min_rounds must be positive")
        if not 0 < self.stop_probability <= 1:
            raise ValueError("stop_probability must be in (0, 1]")
        if self.max_rounds_safety_cap < self.min_rounds:
            raise ValueError("max_rounds_safety_cap cannot be below min_rounds")
        if self.race_prize < 0:
            raise ValueError("race_prize cannot be negative")
        if not 0 <= self.max_private_risk <= 1:
            raise ValueError("max_private_risk must be in [0, 1]")
        if self.history_mode not in {"previous_round", "full"}:
            raise ValueError("history_mode must be 'previous_round' or 'full'")
        if self.run_phase not in {"pilot", "confirmatory"}:
            raise ValueError("run_phase must be 'pilot' or 'confirmatory'")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
        sampling_seed_applied: Optional[bool] = None,
        run_phase: Optional[str] = None,
    ) -> "GameConfig":
        payoffs = data.get("stagePayoffs", {}) or {}
        return cls(
            name=str(data["name"]),
            engine=str(data.get("engine", "ai_race")),
            n_players=int(data.get("nPlayers", 2)),
            safe_progress=float(data.get("safeProgress", 1.0)),
            unsafe_progress=float(data.get("unsafeProgress", 1.5)),
            payoff_safe_safe=float(payoffs.get("safeSafe", 1.0)),
            payoff_safe_unsafe=float(payoffs.get("safeUnsafe", 0.6)),
            payoff_unsafe_safe=float(payoffs.get("unsafeSafe", 2.4)),
            payoff_unsafe_unsafe=float(payoffs.get("unsafeUnsafe", 2.0)),
            min_rounds=int(data.get("minRounds", 5)),
            stop_probability=float(data.get("stopProbability", 0.2)),
            max_rounds_safety_cap=int(data.get("maxRoundsSafetyCap", 100)),
            race_prize=float(data.get("racePrize", 100.0)),
            max_private_risk=float(data.get("maxPrivateRisk", 0.1)),
            history_mode=str(data.get("historyMode", "previous_round")),
            agents_communicate=bool(data.get("agentsCommunicate", False)),
            reveal_opponent_persona_prior=bool(data.get("revealOpponentPersonaPrior", False)),
            language=language or str(data.get("language", "en")),
            prompt_template=str(data.get("promptTemplate", "ai_race_en")),
            prompt_version=str(data.get("promptVersion", "ai-race-paper-v2")),
            run_phase=run_phase or str(data.get("runPhase", "pilot")),
            agents_ref=str(data.get("agents", "companies_default")),
            model=model or str(data.get("model", "LocalQwen")),
            sampling_seed_applied=(
                bool(data.get("samplingSeedApplied", True))
                if sampling_seed_applied is None
                else bool(sampling_seed_applied)
            ),
        )

    def stage_payoff(self, own: Action, opponent: Action) -> float:
        own = Action.coerce(own)
        opponent = Action.coerce(opponent)
        matrix = {
            (Action.SAFE, Action.SAFE): self.payoff_safe_safe,
            (Action.SAFE, Action.UNSAFE): self.payoff_safe_unsafe,
            (Action.UNSAFE, Action.SAFE): self.payoff_unsafe_safe,
            (Action.UNSAFE, Action.UNSAFE): self.payoff_unsafe_unsafe,
        }
        return float(matrix[(own, opponent)])

    def progress_for(self, action: Action) -> float:
        return self.unsafe_progress if Action.coerce(action) is Action.UNSAFE else self.safe_progress

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnRecord:
    """One player decision with the pre- and post-action race state."""

    game_id: str
    model: str
    max_private_risk: float
    prompt_version: str
    run_phase: str
    rep: int
    game_seed: int
    sampling_seed: Optional[int]
    sampling_seed_applied: bool
    round: int
    player: str
    player_index: int
    opponent: str
    action: str
    unsafe: int
    parse_failed: bool
    retry_count: int
    reasoning: str
    raw_response: str
    prompt: str
    own_prev_action: Optional[str]
    opponent_prev_action: Optional[str]
    own_progress_before: float
    opponent_progress_before: float
    progress_gap_before: float
    own_stage_payoff_before: float
    opponent_stage_payoff_before: float
    own_private_risk_before: float
    opponent_private_risk_before: float
    round_payoff: float
    step_increment: float
    own_progress_after: float
    opponent_progress_after: float
    progress_gap_after: float
    cumulative_stage_payoff_after: float
    unsafe_count_after: int
    unsafe_fraction_after: float
    current_private_risk_after: float
    stop_draw: Optional[float]
    stopped: bool
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    logprobs: Optional[dict[str, Any]] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GameResult:
    """Terminal record for one two-player race."""

    game_id: str
    config: dict[str, Any]
    model: str
    max_private_risk: float
    run_phase: str
    rep: int
    game_seed: int
    n_rounds: int
    stop_forced: bool
    player_names: list[str]
    per_round_actions: list[list[str]]
    per_round_payoffs: list[list[float]]
    progress: list[float]
    stage_payoffs: list[float]
    unsafe_counts: list[int]
    unsafe_frequencies: list[float]
    private_risks: list[float]
    outcomes: list[str]
    prizes: list[float]
    setback_eligible: list[bool]
    setback_draws: list[float]
    setbacks: list[bool]
    final_payoffs: list[float]
    stop_draws: list[float] = field(default_factory=list)
    parse_failures: int = 0

    @property
    def tie(self) -> bool:
        return all(outcome == "tie" for outcome in self.outcomes)

    @property
    def winner(self) -> Optional[str]:
        if self.tie:
            return None
        for name, outcome in zip(self.player_names, self.outcomes):
            if outcome == "winner":
                return name
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
