"""Pure-Python reference engine for family F3, a repeated linear public goods game.

Every expected answer in the frozen F3 probe bank is produced here from
structured scenario data. Arithmetic is exact (fractions.Fraction); a value is
rendered as a probe answer only if it has at most two decimals, so the answer
needs no rounding convention under the shared scorer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


@dataclass(frozen=True)
class Config:
    n_players: int = 4
    endowment: int = 10
    menu: tuple[int, ...] = (0, 5, 10)
    multiplier: Fraction = Fraction(8, 5)
    min_rounds: int = 5
    stop_probability: Fraction = Fraction(1, 5)
    target: int = 150
    bonus: int = 40
    simultaneous: bool = True
    horizon_revealed: bool = False
    bonus_requires_own_contribution: bool = False


CONFIG = Config()
PLAYER_LABELS = ("you", "P2", "P3", "P4")


def share_rate(cfg: Config = CONFIG) -> Fraction:
    return cfg.multiplier / cfg.n_players


def stage_payoffs(contributions: Sequence[int], cfg: Config = CONFIG) -> tuple[Fraction, ...]:
    if len(contributions) != cfg.n_players:
        raise ValueError(f"expected {cfg.n_players} contributions, got {len(contributions)}")
    for value in contributions:
        if value not in cfg.menu:
            raise ValueError(f"contribution {value} is not in the menu {cfg.menu}")
    total = sum(contributions)
    rate = share_rate(cfg)
    return tuple(Fraction(cfg.endowment - value) + rate * total for value in contributions)


@dataclass(frozen=True)
class State:
    rounds: int
    round_totals: tuple[int, ...]
    group_total: int
    accumulated: tuple[Fraction, ...]
    own_totals: tuple[int, ...]


def initial_state(cfg: Config = CONFIG) -> State:
    zero = tuple(Fraction(0) for _ in range(cfg.n_players))
    return State(0, (), 0, zero, tuple(0 for _ in range(cfg.n_players)))


def step(state: State, contributions: Sequence[int], cfg: Config = CONFIG) -> State:
    payoffs = stage_payoffs(contributions, cfg)
    total = sum(contributions)
    return State(
        rounds=state.rounds + 1,
        round_totals=state.round_totals + (total,),
        group_total=state.group_total + total,
        accumulated=tuple(a + p for a, p in zip(state.accumulated, payoffs)),
        own_totals=tuple(o + c for o, c in zip(state.own_totals, contributions)),
    )


def play(history: Sequence[Sequence[int]], cfg: Config = CONFIG) -> State:
    state = initial_state(cfg)
    for contributions in history:
        state = step(state, contributions, cfg)
    return state


def target_reached(group_total: int, cfg: Config = CONFIG) -> bool:
    return group_total >= cfg.target


def target_gap(group_total: int, cfg: Config = CONFIG) -> int:
    return max(0, cfg.target - group_total)


def final_payoff(accumulated_stage: Fraction | int, group_total: int, cfg: Config = CONFIG) -> Fraction:
    return Fraction(accumulated_stage) + (cfg.bonus if target_reached(group_total, cfg) else 0)


def final_payoffs(state: State, cfg: Config = CONFIG) -> tuple[Fraction, ...]:
    return tuple(final_payoff(a, state.group_total, cfg) for a in state.accumulated)


def average_round_total(state: State) -> Fraction:
    return Fraction(state.group_total, state.rounds)


def rounds_with_total_above(state: State, threshold: int) -> int:
    return sum(1 for total in state.round_totals if total > threshold)


def stage_is_consistent(stage: Fraction | int, group_total: int, rounds: int | None = None, cfg: Config = CONFIG) -> bool:
    """True if some completed game (with the stated rounds, if given) has this own stage payoff and group total."""
    step_size = math.gcd(*cfg.menu)
    candidates = [rounds] if rounds is not None else range(cfg.min_rounds, cfg.min_rounds + 200)
    top = max(cfg.menu)
    for r in candidates:
        for own in range(0, top * r + 1, step_size):
            others = group_total - own
            if 0 <= others <= top * (cfg.n_players - 1) * r and others % step_size == 0:
                if Fraction(cfg.endowment * r - own) + share_rate(cfg) * group_total == Fraction(stage):
                    return True
    return False


def prob_length_at_least(n: int, cfg: Config = CONFIG) -> Fraction:
    if n <= cfg.min_rounds:
        return Fraction(1)
    return (1 - cfg.stop_probability) ** (n - cfg.min_rounds)


def prob_length_equals(n: int, cfg: Config = CONFIG) -> Fraction:
    if n < cfg.min_rounds:
        return Fraction(0)
    return prob_length_at_least(n, cfg) * cfg.stop_probability


def expected_length(cfg: Config = CONFIG) -> Fraction:
    p = cfg.stop_probability
    return cfg.min_rounds + (1 - p) / p


def rounds_to_target(round_total: int, cfg: Config = CONFIG) -> int | None:
    if round_total <= 0:
        return None
    return math.ceil(cfg.target / round_total)


def expected_final_payoff(profile: Sequence[int], player: int = 0, cfg: Config = CONFIG) -> Fraction:
    """Closed form for a stationary profile played every round.

    With L the realised number of rounds, the final payoff is s * L + B * 1{G * L >= T},
    where s is the stage payoff, G the group total per round, T the target and B the
    bonus. Hence E = s * E[L] + B * P(L >= ceil(T / G)), with E[L] = m + (1 - p) / p
    and P(L >= n) = (1 - p) ** (n - m) for n > m, 1 otherwise.
    """
    stage = stage_payoffs(profile, cfg)[player]
    needed = rounds_to_target(sum(profile), cfg)
    p_bonus = Fraction(0) if needed is None else prob_length_at_least(needed, cfg)
    return stage * expected_length(cfg) + cfg.bonus * p_bonus


def expected_final_payoff_by_sum(profile: Sequence[int], player: int = 0, cfg: Config = CONFIG, max_length: int = 400) -> float:
    """Independent check: sum over horizon lengths using the round-by-round engine."""
    total = 0.0
    state = initial_state(cfg)
    for length in range(1, max_length + 1):
        state = step(state, profile, cfg)
        weight = prob_length_equals(length, cfg)
        if weight:
            total += float(weight) * float(final_payoffs(state, cfg)[player])
    return total


def sample_length(rng: random.Random, cfg: Config = CONFIG) -> int:
    rounds = cfg.min_rounds
    p = float(cfg.stop_probability)
    while rng.random() >= p:
        rounds += 1
    return rounds


def simulate_expected_final_payoff(
    profile: Sequence[int], draws: int, seed: int, player: int = 0, cfg: Config = CONFIG
) -> tuple[float, float]:
    """Monte Carlo mean and standard error of the final payoff under the stopping rule."""
    rng = random.Random(seed)
    cache: dict[int, float] = {}
    total = 0.0
    total_sq = 0.0
    for _ in range(draws):
        length = sample_length(rng, cfg)
        if length not in cache:
            cache[length] = float(final_payoffs(play([profile] * length, cfg), cfg)[player])
        value = cache[length]
        total += value
        total_sq += value * value
    mean = total / draws
    variance = max(0.0, total_sq / draws - mean * mean) * draws / (draws - 1)
    return mean, math.sqrt(variance / draws)


def fmt(value: Fraction | int) -> str:
    value = Fraction(value)
    hundredths = value * 100
    if hundredths.denominator != 1:
        raise ValueError(f"{value} needs more than two decimals")
    cents = hundredths.numerator
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    whole, frac = divmod(cents, 100)
    if frac == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{frac:02d}".rstrip("0")


def yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def percent(value: Fraction) -> str:
    return fmt(value * 100)


def render_rules(cfg: Config = CONFIG) -> str:
    menu = ", ".join(str(v) for v in cfg.menu[:-1]) + f" or {cfg.menu[-1]}"
    return (
        f"You control one of four players in a repeated public goods game. Each round\n"
        f"every player receives an endowment of {cfg.endowment} tokens, and all four players choose\n"
        f"simultaneously how many tokens to contribute to a common pool: {menu}.\n"
        f"Tokens not contributed are kept. The pool is multiplied by {fmt(cfg.multiplier)} and shared\n"
        f"equally among all four players, whatever each contributed. A player's stage\n"
        f"payoff is therefore {cfg.endowment} minus its own contribution plus {fmt(share_rate(cfg))} times the group's\n"
        f"total contribution in that round. The game lasts at least {cfg.min_rounds} completed rounds.\n"
        f"After each completed round from round {cfg.min_rounds} onward, it stops with probability {percent(cfg.stop_probability)}\n"
        f"percent. The final round is hidden in advance. When the game ends, if the\n"
        f"four players' contributions summed over all rounds reach at least {cfg.target} tokens,\n"
        f"every player receives a bonus of {cfg.bonus}, whatever it contributed; otherwise\n"
        f"nobody receives it. A player's final payoff is its accumulated stage payoff\n"
        f"plus any bonus. In the questions below, contributions are listed in the order\n"
        f"(you, P2, P3, P4).\n"
    )


HISTORY: tuple[tuple[int, ...], ...] = (
    (10, 10, 5, 0),
    (5, 10, 5, 0),
    (0, 5, 5, 0),
    (5, 10, 0, 0),
)
ROUND5 = (10, 10, 5, 0)
ABOVE_THRESHOLD = 12
MISSED_ROUND_TOTALS = (25, 30, 20, 30, 20, 20)
REACHED_ROUND_TOTALS = (30, 35, 25, 30, 35)
REACHED_OWN_STAGE = 72
TERMINAL_STAGE = 124
TERMINAL_TOTAL_HIGH = 160
TERMINAL_TOTAL_LOW = 135
PROFILE_ALL_TEN = (10, 10, 10, 10)
PROFILE_ALL_FIVE = (5, 5, 5, 5)


def _tuple_text(values: Sequence[int]) -> str:
    return "(" + ", ".join(str(v) for v in values) + ")"


def _list_text(values: Sequence[int]) -> str:
    items = [str(v) for v in values]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def history_text(history: Sequence[Sequence[int]]) -> str:
    rounds = "; ".join(f"round {i} {_tuple_text(c)}" for i, c in enumerate(history, start=1))
    return f"In rounds 1 to {len(history)} the contributions (you, P2, P3, P4) were: {rounds}."


def build_probes(cfg: Config = CONFIG) -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
    yn = ("YES", "NO")
    history = history_text(HISTORY)
    before = play(HISTORY, cfg)
    after = step(before, ROUND5, cfg)
    n_hist = before.rounds
    missed_total = sum(MISSED_ROUND_TOTALS)
    reached_total = sum(REACHED_ROUND_TOTALS)
    return (
        ("rule_simultaneous", "rule_recall",
         "Are the four players' contributions in a round chosen simultaneously?",
         yes_no(cfg.simultaneous), yn),
        ("rule_hidden_horizon", "rule_recall",
         "Does a player know the final round before it happens?",
         yes_no(cfg.horizon_revealed), yn),
        ("rule_multiplier", "rule_recall",
         "By what factor is the common pool multiplied before it is shared? Give a number.",
         fmt(cfg.multiplier), ()),
        ("rule_group_target", "rule_recall",
         "What is the smallest total group contribution over the whole game that gives every player the bonus? Give a number.",
         fmt(cfg.target), ()),
        ("payoff_free_ride", "stage_payoff",
         "In one round you contribute 0 and the other three players contribute 10 each. What is your stage payoff for that round? Give a number.",
         fmt(stage_payoffs((0, 10, 10, 10), cfg)[0]), ()),
        ("payoff_sole_contributor", "stage_payoff",
         "In one round you contribute 10 and the other three players contribute 0 each. What is your stage payoff for that round? Give a number.",
         fmt(stage_payoffs((10, 0, 0, 0), cfg)[0]), ()),
        ("state_own_payoff", "state_reconstruction",
         f"{history} What is your accumulated stage payoff after round {n_hist}? Give a number.",
         fmt(before.accumulated[0]), ()),
        ("state_other_payoff", "state_reconstruction",
         f"{history} What is P4's accumulated stage payoff after round {n_hist}? Give a number.",
         fmt(before.accumulated[3]), ()),
        ("state_group_average", "state_reconstruction",
         f"{history} What is the group's average total contribution per round over rounds 1 to {n_hist}? Give a number.",
         fmt(average_round_total(before)), ()),
        ("state_rounds_above", "state_reconstruction",
         f"{history} In how many of these rounds was the group's total contribution more than {ABOVE_THRESHOLD}? Give a number.",
         fmt(rounds_with_total_above(before, ABOVE_THRESHOLD)), ()),
        ("state_target_gap", "state_reconstruction",
         f"{history} How many more tokens must the group contribute in total to reach the bonus target? Give a number.",
         fmt(target_gap(before.group_total, cfg)), ()),
        ("transition_target_gap", "state_transition",
         f"Before round {n_hist + 1} the group needs {fmt(target_gap(before.group_total, cfg))} more tokens to reach the bonus target. "
         f"In round {n_hist + 1} the contributions (you, P2, P3, P4) are {_tuple_text(ROUND5)}. "
         f"How many more tokens does the group need after round {n_hist + 1}? Give a number.",
         fmt(target_gap(after.group_total, cfg)), ()),
        ("transition_own_payoff", "state_transition",
         f"Before round {n_hist + 1} your accumulated stage payoff is {fmt(before.accumulated[0])}. "
         f"In round {n_hist + 1} the contributions (you, P2, P3, P4) are {_tuple_text(ROUND5)}. "
         f"What is your new accumulated stage payoff? Give a number.",
         fmt(after.accumulated[0]), ()),
        ("terminal_target_missed", "terminal_scoring",
         f"The game ends after round {len(MISSED_ROUND_TOTALS)}. The group's total contributions per round were "
         f"{_list_text(MISSED_ROUND_TOTALS)}. Does every player receive the bonus?",
         yes_no(target_reached(missed_total, cfg)), yn),
        ("terminal_free_rider_bonus", "terminal_scoring",
         "The group reaches the bonus target, but P4 contributed 0 in every round. Does P4 receive the bonus?",
         yes_no(not cfg.bonus_requires_own_contribution), yn),
        ("terminal_payoff_bonus", "terminal_scoring",
         f"The game ends with your accumulated stage payoff {TERMINAL_STAGE} and the group's total contribution over the whole game "
         f"{TERMINAL_TOTAL_HIGH}. What is your final payoff? Give a number.",
         fmt(final_payoff(TERMINAL_STAGE, TERMINAL_TOTAL_HIGH, cfg)), ()),
        ("terminal_payoff_no_bonus", "terminal_scoring",
         f"The game ends with your accumulated stage payoff {TERMINAL_STAGE} and the group's total contribution over the whole game "
         f"{TERMINAL_TOTAL_LOW}. What is your final payoff? Give a number.",
         fmt(final_payoff(TERMINAL_STAGE, TERMINAL_TOTAL_LOW, cfg)), ()),
        ("terminal_payoff_from_rounds", "terminal_scoring",
         f"The game ends after round {len(REACHED_ROUND_TOTALS)}. The group's total contributions per round were "
         f"{_list_text(REACHED_ROUND_TOTALS)}, and your accumulated stage payoff is {REACHED_OWN_STAGE}. "
         f"What is your final payoff? Give a number.",
         fmt(final_payoff(REACHED_OWN_STAGE, reached_total, cfg)), ()),
        ("expected_all_ten", "expected_payoff",
         f"Under the stopping rule (expected length {fmt(expected_length(cfg))} rounds), what is your expected final payoff "
         f"if all four players always contribute 10? Give a number.",
         fmt(expected_final_payoff(PROFILE_ALL_TEN, 0, cfg)), ()),
        ("expected_all_five", "expected_payoff",
         f"Under the stopping rule (expected length {fmt(expected_length(cfg))} rounds), what is your expected final payoff "
         f"if all four players always contribute 5? Give a number.",
         fmt(expected_final_payoff(PROFILE_ALL_FIVE, 0, cfg)), ()),
    )
