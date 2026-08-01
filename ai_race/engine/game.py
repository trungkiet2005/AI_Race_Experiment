"""Stepwise, simultaneous two-player AI Race engine."""
from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from .agent import RaceAgent
from .round import (
    AIRaceRound,
    extract_reasoning,
    parse_action,
    raw_response_text,
)
from .scoring import joint_round_outcome, terminal_scoring
from .state import Action, GameConfig, GameResult, TurnRecord

_SEED_MOD = 2_147_483_647
_SAMPLING_SEED_STRIDE = 100_000
_RETRY_SEED_STRIDE = 10_000_019
_HORIZON_STREAM = 17
_SETBACK_STREAM = 29


def _stream_seed(base_seed: int, stream: int) -> int:
    """Derive stable, disjoint RNG streams without Python's salted ``hash``."""
    return (int(base_seed) * 1_000_003 + int(stream) * 97_409 + 11_729) % _SEED_MOD


def _response_logprobs(response: str | dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(response, dict):
        return None
    details = {
        key: response[key]
        for key in ("token_logprobs", "cumulative_logprob", "top_alternatives")
        if key in response
    }
    return details or None


class AIRaceGame:
    """One race with two model-controlled companies.

    The game exposes a stepwise API so many races can be evaluated in lockstep by a
    batched model backend. Every prompt within a round is rendered before either
    response is applied.
    """

    def __init__(
        self,
        config: GameConfig,
        agents: Sequence[RaceAgent],
        *,
        template: str,
        game_id: str,
        seed: int,
        rep: int = 0,
    ):
        if len(agents) != 2:
            raise ValueError("AIRaceGame requires exactly two agents")
        self.config = config
        self.agents = list(agents)
        self.template = str(template)
        self.game_id = str(game_id)
        self.seed = int(seed)
        self.rep = int(rep)

        self.history: list[dict[str, Any]] = []
        self.turns: list[TurnRecord] = []
        self.progress = [0.0, 0.0]
        self.stage_payoffs = [0.0, 0.0]
        self.unsafe_counts = [0, 0]
        self.stop_draws: list[float] = []
        self.stop_forced = False

        self._horizon_rng = random.Random(_stream_seed(self.seed, _HORIZON_STREAM))
        self._finished = False
        self._result: Optional[GameResult] = None

    @property
    def current_round(self) -> int:
        return len(self.history) + 1

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def result(self) -> Optional[GameResult]:
        return self._result

    def sampling_seed(self, player_index: int, round_number: int, retry: int = 0) -> int:
        if player_index not in (0, 1):
            raise ValueError("player_index must be 0 or 1")
        seed = (
            self.seed * _SAMPLING_SEED_STRIDE
            + int(round_number) * 100
            + int(player_index)
            + int(retry) * _RETRY_SEED_STRIDE
        )
        return seed % _SEED_MOD

    def build_round_prompts(self) -> list[str]:
        if self._finished:
            raise RuntimeError("Cannot build prompts for a finished race")
        return AIRaceRound(self, self.current_round).build_prompts()

    def _should_stop_after_round(self, round_number: int) -> tuple[Optional[float], bool]:
        if round_number < self.config.min_rounds:
            return None, False
        draw = self._horizon_rng.random()
        self.stop_draws.append(draw)
        stochastic_stop = draw < self.config.stop_probability
        forced_stop = round_number >= self.config.max_rounds_safety_cap
        if forced_stop and not stochastic_stop:
            self.stop_forced = True
        return draw, stochastic_stop or forced_stop

    def apply_round_responses(
        self,
        responses: Sequence[str | dict[str, Any]],
        *,
        prompts: Optional[Sequence[str]] = None,
        retry_counts: Optional[Sequence[int]] = None,
        attempt_histories: Optional[Sequence[Sequence[dict[str, Any]]]] = None,
    ) -> Optional[GameResult]:
        """Apply two responses simultaneously and optionally return a terminal result."""
        if self._finished:
            raise RuntimeError("Cannot apply responses to a finished race")
        if len(responses) != 2:
            raise ValueError("Exactly two responses are required")
        if prompts is None:
            prompts = self.build_round_prompts()
        if len(prompts) != 2:
            raise ValueError("Exactly two prompts are required")
        retry_counts = list(retry_counts or (0, 0))
        if len(retry_counts) != 2:
            raise ValueError("Exactly two retry counts are required")
        if attempt_histories is None:
            attempt_histories = [
                [
                    {
                        "attempt": int(retry_counts[player_index]),
                        "sampling_seed": (
                            self.sampling_seed(
                                player_index,
                                self.current_round,
                                int(retry_counts[player_index]),
                            )
                            if self.config.sampling_seed_applied
                            else None
                        ),
                        "sampling_seed_applied": self.config.sampling_seed_applied,
                        "raw_response": raw_response_text(responses[player_index]),
                        "parse_failed": bool(parse_action(responses[player_index])[1]),
                    }
                ]
                for player_index in (0, 1)
            ]
        attempt_histories = [list(history) for history in attempt_histories]
        if len(attempt_histories) != 2:
            raise ValueError("Exactly two attempt histories are required")

        round_number = self.current_round
        progress_before = list(self.progress)
        stage_before = list(self.stage_payoffs)
        unsafe_before = list(self.unsafe_counts)
        completed_rounds_before = len(self.history)
        private_risks_before = [
            (
                self.config.max_private_risk
                * unsafe_before[player_index]
                / completed_rounds_before
                if completed_rounds_before
                else 0.0
            )
            for player_index in (0, 1)
        ]
        previous_actions = (
            [Action.coerce(action) for action in self.history[-1]["actions"]]
            if self.history
            else [None, None]
        )

        parsed = [parse_action(response) for response in responses]
        actions = [item[0] for item in parsed]
        parse_failed = [item[1] for item in parsed]
        payoffs, increments = joint_round_outcome(self.config, actions)

        for player_index in (0, 1):
            self.progress[player_index] += increments[player_index]
            self.stage_payoffs[player_index] += payoffs[player_index]
            self.unsafe_counts[player_index] += actions[player_index].unsafe

        stop_draw, stopped = self._should_stop_after_round(round_number)
        round_record = {
            "round": round_number,
            "actions": [action.value for action in actions],
            "payoffs": list(payoffs),
            "increments": list(increments),
            "progress_after": list(self.progress),
            "stage_payoffs_after": list(self.stage_payoffs),
            "unsafe_counts_after": list(self.unsafe_counts),
            "stop_draw": stop_draw,
            "stopped": stopped,
        }
        self.history.append(round_record)

        names = [agent.name for agent in self.agents]
        for player_index in (0, 1):
            opponent_index = 1 - player_index
            rounds_played = len(self.history)
            unsafe_fraction = self.unsafe_counts[player_index] / rounds_played
            current_risk = self.config.max_private_risk * unsafe_fraction
            retry_count = int(retry_counts[player_index])
            self.turns.append(
                TurnRecord(
                    game_id=self.game_id,
                    model=self.config.model,
                    max_private_risk=self.config.max_private_risk,
                    prompt_version=self.config.prompt_version,
                    run_phase=self.config.run_phase,
                    persona_condition=self.config.persona_condition,
                    seat_persona_role=self.agents[player_index].persona_role,
                    rep=self.rep,
                    game_seed=self.seed,
                    sampling_seed=(
                        self.sampling_seed(
                            player_index,
                            round_number,
                            retry_count,
                        )
                        if self.config.sampling_seed_applied
                        else None
                    ),
                    sampling_seed_applied=self.config.sampling_seed_applied,
                    round=round_number,
                    player=names[player_index],
                    player_index=player_index,
                    opponent=names[opponent_index],
                    action=actions[player_index].value,
                    unsafe=actions[player_index].unsafe,
                    parse_failed=bool(parse_failed[player_index]),
                    retry_count=retry_count,
                    reasoning=extract_reasoning(responses[player_index]),
                    raw_response=raw_response_text(responses[player_index]),
                    prompt=str(prompts[player_index]),
                    own_prev_action=(
                        previous_actions[player_index].value
                        if previous_actions[player_index] is not None
                        else None
                    ),
                    opponent_prev_action=(
                        previous_actions[opponent_index].value
                        if previous_actions[opponent_index] is not None
                        else None
                    ),
                    own_progress_before=progress_before[player_index],
                    opponent_progress_before=progress_before[opponent_index],
                    progress_gap_before=(
                        progress_before[player_index] - progress_before[opponent_index]
                    ),
                    own_stage_payoff_before=stage_before[player_index],
                    opponent_stage_payoff_before=stage_before[opponent_index],
                    own_private_risk_before=private_risks_before[player_index],
                    opponent_private_risk_before=private_risks_before[opponent_index],
                    round_payoff=payoffs[player_index],
                    step_increment=increments[player_index],
                    own_progress_after=self.progress[player_index],
                    opponent_progress_after=self.progress[opponent_index],
                    progress_gap_after=(
                        self.progress[player_index] - self.progress[opponent_index]
                    ),
                    cumulative_stage_payoff_after=self.stage_payoffs[player_index],
                    unsafe_count_after=self.unsafe_counts[player_index],
                    unsafe_fraction_after=unsafe_fraction,
                    current_private_risk_after=current_risk,
                    stop_draw=stop_draw,
                    stopped=stopped,
                    attempt_history=attempt_histories[player_index],
                    logprobs=_response_logprobs(responses[player_index]),
                )
            )

        if stopped:
            self._finished = True
            self._result = self._finalize()
        return self._result

    def _finalize(self) -> GameResult:
        rounds_played = len(self.history)
        setback_rng = random.Random(_stream_seed(self.seed, _SETBACK_STREAM))
        fixed_seat_draws = [setback_rng.random(), setback_rng.random()]
        terminal = terminal_scoring(
            stage_payoffs=self.stage_payoffs,
            progress=self.progress,
            unsafe_counts=self.unsafe_counts,
            rounds_played=rounds_played,
            max_private_risk=self.config.max_private_risk,
            race_prize=self.config.race_prize,
            setback_draws=fixed_seat_draws,
        )
        unsafe_frequencies = [
            count / rounds_played if rounds_played else 0.0
            for count in self.unsafe_counts
        ]
        return GameResult(
            game_id=self.game_id,
            config=self.config.snapshot(),
            model=self.config.model,
            max_private_risk=self.config.max_private_risk,
            run_phase=self.config.run_phase,
            persona_condition=self.config.persona_condition,
            persona_roles=[agent.persona_role for agent in self.agents],
            rep=self.rep,
            game_seed=self.seed,
            n_rounds=rounds_played,
            stop_forced=self.stop_forced,
            player_names=[agent.name for agent in self.agents],
            per_round_actions=[list(record["actions"]) for record in self.history],
            per_round_payoffs=[list(record["payoffs"]) for record in self.history],
            progress=list(self.progress),
            stage_payoffs=list(self.stage_payoffs),
            unsafe_counts=list(self.unsafe_counts),
            unsafe_frequencies=unsafe_frequencies,
            private_risks=terminal.private_risks,
            outcomes=terminal.outcomes,
            prizes=terminal.prizes,
            setback_eligible=terminal.setback_eligible,
            setback_draws=terminal.setback_draws,
            setbacks=terminal.setbacks,
            final_payoffs=terminal.final_payoffs,
            stop_draws=list(self.stop_draws),
            parse_failures=sum(int(turn.parse_failed) for turn in self.turns),
        )
