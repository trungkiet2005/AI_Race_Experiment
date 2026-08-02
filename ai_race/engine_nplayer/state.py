"""Typed configuration and output records for the N-player AI Race.

Mirrors ``ai_race.engine.state`` in shape, generalised from exactly two seats
to any ``n_players >= 2``. ``Action`` is reused unchanged from the two-player
engine rather than redefined here -- SAFE/UNSAFE is the same concept in both
mechanisms, only the payoff and outcome rules differ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Optional

from ai_race.engine.state import Action

__all__ = ["Action", "NPlayerGameConfig", "NPlayerTurnRecord", "NPlayerGameResult"]


@dataclass
class NPlayerGameConfig:
    """Game mechanism loaded from ``configs/game_nplayer/*.json``.

    The payoff is parametric (``cost``, ``benefit``, ``speed``) rather than a
    fixed 2x2 matrix, following Appendix B of the reference paper
    (``references/papers/sources/jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.md``, "N-player AI
    Race Definition") with the found-out term ``p_fo`` dropped. At
    ``n_players=2`` with the defaults below (cost=1, benefit=4, speed=1.5) the
    formula reproduces the shipped two-player payoff matrix
    (1.0 / 0.6 / 2.4 / 2.0) exactly -- see
    ``ai_race/tests/test_nplayer_scoring.py`` for the cross-check.
    """

    name: str
    engine: str = "ai_race_nplayer"
    n_players: int = 3
    safe_progress: float = 1.0
    # Also the payoff weight of an Unsafe player relative to a Safe one, per
    # the paper's formula -- one number does both jobs, matching the source.
    speed: float = 1.5
    cost: float = 1.0
    benefit: float = 4.0
    min_rounds: int = 5
    stop_probability: float = 0.2
    max_rounds_safety_cap: int = 100
    race_prize: float = 100.0
    max_private_risk: float = 0.1
    history_mode: str = "previous_round"
    language: str = "en"
    prompt_template: str = "ai_race_nplayer_en"
    prompt_version: str = "ai-race-nplayer-v1"
    run_phase: str = "pilot"
    agents_ref: str = "companies_nplayer_default"
    persona_condition: str = "none"
    persona_sha256: str = ""
    model: str = "LocalQwen"
    sampling_seed_applied: bool = True

    def __post_init__(self) -> None:
        if self.engine != "ai_race_nplayer":
            raise ValueError(
                f"Unsupported engine {self.engine!r}; expected 'ai_race_nplayer'"
            )
        if self.n_players < 2:
            raise ValueError("The N-player AI Race requires at least two players")
        mechanism_values = {
            "safe_progress": self.safe_progress,
            "speed": self.speed,
            "cost": self.cost,
            "benefit": self.benefit,
            "race_prize": self.race_prize,
            "max_private_risk": self.max_private_risk,
            "stop_probability": self.stop_probability,
        }
        nonfinite = [
            key for key, value in mechanism_values.items()
            if not math.isfinite(float(value))
        ]
        if nonfinite:
            raise ValueError(f"Game mechanism values must be finite: {nonfinite}")
        if self.safe_progress <= 0:
            raise ValueError("safe_progress must be positive")
        if self.speed <= 1.0:
            raise ValueError("speed must be greater than 1 (Unsafe must be faster)")
        if self.benefit <= 0:
            raise ValueError("benefit must be positive")
        if self.cost < 0:
            raise ValueError("cost cannot be negative")
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
        if self.history_mode not in {"previous_round", "first_and_previous", "full"}:
            raise ValueError(
                "history_mode must be 'previous_round', 'first_and_previous', "
                "or 'full'"
            )
        if self.run_phase not in {"pilot", "confirmatory"}:
            raise ValueError("run_phase must be 'pilot' or 'confirmatory'")
        if not str(self.persona_condition).strip():
            raise ValueError(
                "persona_condition must be a non-empty label; use 'none' for the "
                "neutral baseline so an unlabelled run cannot be pooled with it"
            )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
        sampling_seed_applied: Optional[bool] = None,
        run_phase: Optional[str] = None,
        persona_condition: Optional[str] = None,
        persona_sha256: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> "NPlayerGameConfig":
        return cls(
            name=str(data["name"]),
            engine=str(data.get("engine", "ai_race_nplayer")),
            n_players=int(data.get("nPlayers", 3)),
            safe_progress=float(data.get("safeProgress", 1.0)),
            speed=float(data.get("speed", 1.5)),
            cost=float(data.get("cost", 1.0)),
            benefit=float(data.get("benefit", 4.0)),
            min_rounds=int(data.get("minRounds", 5)),
            stop_probability=float(data.get("stopProbability", 0.2)),
            max_rounds_safety_cap=int(data.get("maxRoundsSafetyCap", 100)),
            race_prize=float(data.get("racePrize", 100.0)),
            max_private_risk=float(data.get("maxPrivateRisk", 0.1)),
            history_mode=str(data.get("historyMode", "previous_round")),
            language=language or str(data.get("language", "en")),
            prompt_template=str(data.get("promptTemplate", "ai_race_nplayer_en")),
            prompt_version=prompt_version
            or str(data.get("promptVersion", "ai-race-nplayer-v1")),
            run_phase=run_phase or str(data.get("runPhase", "pilot")),
            agents_ref=str(data.get("agents", "companies_nplayer_default")),
            persona_condition=str(persona_condition or "none"),
            persona_sha256=str(persona_sha256 or ""),
            model=model or str(data.get("model", "LocalQwen")),
            sampling_seed_applied=(
                bool(data.get("samplingSeedApplied", True))
                if sampling_seed_applied is None
                else bool(sampling_seed_applied)
            ),
        )

    def unsafe_progress(self) -> float:
        return self.safe_progress * self.speed

    def progress_for(self, action: Action) -> float:
        return (
            self.unsafe_progress()
            if Action.coerce(action) is Action.UNSAFE
            else self.safe_progress
        )

    def _share_weight(self, k_safe: int) -> float:
        """Denominator ``k + speed*(N-k)`` shared by both payoff branches."""
        n_unsafe = self.n_players - k_safe
        return float(k_safe) + self.speed * float(n_unsafe)

    def stage_payoff_safe(self, k_safe: int) -> float:
        """Payoff to one SAFE player when ``k_safe`` of ``n_players`` chose SAFE.

        ``k_safe`` must include the player asking (so ``1 <= k_safe <=
        n_players``); the ``k_safe == n_players`` branch (everyone SAFE) is
        the paper's special case where the shared benefit is split evenly
        rather than weighted by speed, since there is no Unsafe player to
        weight against.
        """
        if not 1 <= k_safe <= self.n_players:
            raise ValueError("k_safe must be between 1 and n_players inclusive")
        if k_safe == self.n_players:
            return -self.cost + self.benefit / self.n_players
        return -self.cost + self.benefit / self._share_weight(k_safe)

    def stage_payoff_unsafe(self, k_safe: int) -> float:
        """Payoff to one UNSAFE player when ``k_safe`` of ``n_players`` chose SAFE.

        ``k_safe`` is the count of *other* SAFE players plus any other SAFE
        players in the group; the asking player is Unsafe by definition, so
        ``0 <= k_safe < n_players``.
        """
        if not 0 <= k_safe < self.n_players:
            raise ValueError("k_safe must be between 0 and n_players - 1 inclusive")
        return self.speed * self.benefit / self._share_weight(k_safe)

    def stage_payoff_for(self, action: Action, k_safe: int) -> float:
        """Payoff to one player given their own action and the group's SAFE count."""
        if Action.coerce(action) is Action.SAFE:
            return self.stage_payoff_safe(k_safe)
        return self.stage_payoff_unsafe(k_safe)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NPlayerTurnRecord:
    """One player decision with the pre- and post-action race state.

    Generalises ``ai_race.engine.state.TurnRecord``'s singular ``opponent_*``
    fields to lists over the ``n_players - 1`` co-players, ordered by their
    ``player_index``. ``*_gap_*`` fields report the gap to the *leader* among
    co-players (``own - max(others)``) since with more than one co-player
    there is no single "the opponent" to diff against.
    """

    game_id: str
    model: str
    n_players: int
    max_private_risk: float
    prompt_version: str
    run_phase: str
    persona_condition: str
    seat_persona_role: str
    rep: int
    game_seed: int
    sampling_seed: Optional[int]
    sampling_seed_applied: bool
    round: int
    player: str
    player_index: int
    others: list[str]
    action: str
    unsafe: int
    parse_failed: bool
    retry_count: int
    reasoning: str
    raw_response: str
    prompt: str
    own_prev_action: Optional[str]
    others_prev_actions: list[Optional[str]]
    own_progress_before: float
    others_progress_before: list[float]
    progress_gap_before: float
    own_stage_payoff_before: float
    others_stage_payoff_before: list[float]
    own_private_risk_before: float
    others_private_risk_before: list[float]
    round_payoff: float
    step_increment: float
    own_progress_after: float
    others_progress_after: list[float]
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
class NPlayerGameResult:
    """Terminal record for one N-player race."""

    game_id: str
    config: dict[str, Any]
    model: str
    max_private_risk: float
    run_phase: str
    persona_condition: str
    persona_roles: list[str]
    rep: int
    game_seed: int
    n_players: int
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
    def is_full_tie(self) -> bool:
        return all(outcome == "tie" for outcome in self.outcomes)

    @property
    def winners(self) -> list[str]:
        """Names of every prize recipient (the sole winner, or every tied leader)."""
        return [
            name
            for name, outcome in zip(self.player_names, self.outcomes)
            if outcome in {"winner", "tie"}
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
