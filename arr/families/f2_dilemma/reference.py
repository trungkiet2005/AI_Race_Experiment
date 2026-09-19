"""Pure-Python reference engine for task family F2, a repeated Prisoner's Dilemma.

Every expected answer in the F2 probe bank is derived here from the stated
history. All arithmetic uses exact fractions so that an answer is never the
product of a floating-point rounding convention.
"""

from __future__ import annotations

import random
from fractions import Fraction

COOPERATE = "COOPERATE"
DEFECT = "DEFECT"
ACTIONS = (COOPERATE, DEFECT)

SIMULTANEOUS = True
HORIZON_KNOWN_IN_ADVANCE = False
MIN_ROUNDS = 5
STOP_PROBABILITY = Fraction(1, 5)
BONUS_THRESHOLD = Fraction(3, 5)
BONUS = Fraction("7.5")

PAYOFF = {
    (COOPERATE, COOPERATE): Fraction("3.2"),
    (COOPERATE, DEFECT): Fraction("0.4"),
    (DEFECT, COOPERATE): Fraction("4.7"),
    (DEFECT, DEFECT): Fraction("1.3"),
}

C, D = COOPERATE, DEFECT

HISTORY_A_OWN = (C, C, D, D, C, C)
HISTORY_A_OPP = (C, D, D, C, D, C)
HISTORY_B_OWN = (C, D, C, C, C)
HISTORY_B_OPP = (D, C, D, D, C)
HISTORY_C_OPP = (C, D, C, C, D, C, D)


def fmt(value: Fraction | int | bool) -> str:
    """Render an answer as the bank stores it: YES/NO, an integer, or <= 2 decimals."""
    if isinstance(value, bool):
        return "YES" if value else "NO"
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    scaled = value * 100
    if scaled.denominator != 1:
        raise ValueError(f"{value} needs more than two decimals")
    sign = "-" if scaled < 0 else ""
    cents = abs(scaled.numerator)
    text = f"{cents // 100}.{cents % 100:02d}".rstrip("0")
    return sign + text


def percent(value: Fraction) -> Fraction:
    return Fraction(value) * 100


def actions_phrase(actions: tuple[str, ...]) -> str:
    return ", ".join(actions)


def check_history(own: tuple[str, ...], opp: tuple[str, ...] | None = None) -> None:
    if any(a not in ACTIONS for a in own):
        raise ValueError(f"unknown action in {own}")
    if opp is not None:
        if any(a not in ACTIONS for a in opp):
            raise ValueError(f"unknown action in {opp}")
        if len(own) != len(opp):
            raise ValueError("histories must have equal length")


def stage_payoff(own: str, opp: str) -> Fraction:
    return PAYOFF[(own, opp)]


def accumulated_payoffs(own: tuple[str, ...], opp: tuple[str, ...]) -> tuple[Fraction, Fraction]:
    check_history(own, opp)
    mine = sum((stage_payoff(a, b) for a, b in zip(own, opp)), Fraction(0))
    theirs = sum((stage_payoff(b, a) for a, b in zip(own, opp)), Fraction(0))
    return mine, theirs


def count(actions: tuple[str, ...], action: str) -> int:
    check_history(actions)
    return sum(1 for a in actions if a == action)


def cooperation_rate(actions: tuple[str, ...]) -> Fraction:
    check_history(actions)
    return Fraction(count(actions, COOPERATE), len(actions))


def last_round_of(actions: tuple[str, ...], action: str) -> int:
    check_history(actions)
    rounds = [i + 1 for i, a in enumerate(actions) if a == action]
    if not rounds:
        raise ValueError(f"{action} never played")
    return rounds[-1]


def bonus_eligible(cooperations: int, rounds: int) -> bool:
    if rounds < 1 or not 0 <= cooperations <= rounds:
        raise ValueError("invalid cooperation count")
    return Fraction(cooperations, rounds) >= BONUS_THRESHOLD


def final_payoff(stage_total: Fraction, cooperations: int, rounds: int) -> Fraction:
    return Fraction(stage_total) + (BONUS if bonus_eligible(cooperations, rounds) else 0)


def final_payoffs(own: tuple[str, ...], opp: tuple[str, ...]) -> tuple[Fraction, Fraction]:
    mine, theirs = accumulated_payoffs(own, opp)
    rounds = len(own)
    return (
        final_payoff(mine, count(own, COOPERATE), rounds),
        final_payoff(theirs, count(opp, COOPERATE), rounds),
    )


def transition_payoff(before: Fraction, own: str, opp: str) -> Fraction:
    return Fraction(before) + stage_payoff(own, opp)


def transition_cooperation_rate(cooperations_before: int, rounds_before: int, action: str) -> Fraction:
    return Fraction(cooperations_before + (action == COOPERATE), rounds_before + 1)


def prob_length_at_least(k: int) -> Fraction:
    """P(N >= k): rounds 1..MIN_ROUNDS always happen, then each continuation survives w.p. 1 - p."""
    if k <= MIN_ROUNDS:
        return Fraction(1)
    return (1 - STOP_PROBABILITY) ** (k - MIN_ROUNDS)


def prob_length_equal(n: int) -> Fraction:
    if n < MIN_ROUNDS:
        return Fraction(0)
    return prob_length_at_least(n) * STOP_PROBABILITY


def expected_length() -> Fraction:
    """E[N] = MIN_ROUNDS + (1 - p) / p, the mean of MIN_ROUNDS plus a geometric count of continuations."""
    return MIN_ROUNDS + (1 - STOP_PROBABILITY) / STOP_PROBABILITY


def _strategy_action(prefix: tuple[str, ...], tail: str, k: int) -> str:
    return prefix[k - 1] if k <= len(prefix) else tail


def expected_final_payoff(
    own_prefix: tuple[str, ...], own_tail: str, opp_prefix: tuple[str, ...], opp_tail: str
) -> Fraction:
    """Closed-form E[final payoff] for open-loop strategies: a fixed prefix, then a constant action.

    Stage part: E[sum_{k<=N} s_k] = sum_k P(N>=k) s_k, where the constant tail contributes
    s_tail * (E[N] - sum_{k<=K} P(N>=k)).
    Bonus part: after round K the cooperation rate (c_K + t (n - K)) / n is monotone in n,
    so eligibility changes at most once; it is summed exactly up to the round where it
    reaches its limit and the remaining mass is P(N >= M).
    """
    check_history(own_prefix)
    check_history(opp_prefix)
    horizon = max(len(own_prefix), len(opp_prefix))
    stage = Fraction(0)
    covered = Fraction(0)
    for k in range(1, horizon + 1):
        own = _strategy_action(own_prefix, own_tail, k)
        opp = _strategy_action(opp_prefix, opp_tail, k)
        stage += prob_length_at_least(k) * stage_payoff(own, opp)
        covered += prob_length_at_least(k)
    stage += stage_payoff(own_tail, opp_tail) * (expected_length() - covered)

    def coop_count(n: int) -> int:
        return sum(1 for k in range(1, n + 1) if _strategy_action(own_prefix, own_tail, k) == COOPERATE)

    limit = own_tail == COOPERATE and Fraction(1) >= BONUS_THRESHOLD
    bonus_mass = Fraction(0)
    n = MIN_ROUNDS
    while True:
        eligible = bonus_eligible(coop_count(n), n)
        if n >= max(horizon, MIN_ROUNDS) and eligible == limit:
            bonus_mass += prob_length_at_least(n) * (1 if limit else 0)
            break
        bonus_mass += prob_length_equal(n) * (1 if eligible else 0)
        n += 1
    return stage + BONUS * bonus_mass


def sample_length(rng: random.Random) -> int:
    n = MIN_ROUNDS
    while rng.random() >= float(STOP_PROBABILITY):
        n += 1
    return n


def play(own_prefix: tuple[str, ...], own_tail: str, opp_prefix: tuple[str, ...], opp_tail: str, rounds: int):
    own = tuple(_strategy_action(own_prefix, own_tail, k) for k in range(1, rounds + 1))
    opp = tuple(_strategy_action(opp_prefix, opp_tail, k) for k in range(1, rounds + 1))
    return final_payoffs(own, opp)


def derive_answers() -> dict[str, str]:
    """Expected answer string for every F2 probe, computed from the stated inputs."""
    own_a, opp_a = accumulated_payoffs(HISTORY_A_OWN, HISTORY_A_OPP)
    final_b = final_payoffs(HISTORY_B_OWN, HISTORY_B_OPP)
    final_a = final_payoffs(HISTORY_A_OWN, HISTORY_A_OPP)
    return {
        "rule_simultaneous": fmt(SIMULTANEOUS),
        "rule_hidden_horizon": fmt(HORIZON_KNOWN_IN_ADVANCE),
        "rule_min_rounds": fmt(MIN_ROUNDS),
        "rule_bonus_threshold": fmt(percent(BONUS_THRESHOLD)),
        "payoff_coop_defect": fmt(stage_payoff(COOPERATE, DEFECT)),
        "payoff_defect_coop": fmt(stage_payoff(DEFECT, COOPERATE)),
        "state_opp_defections": fmt(count(HISTORY_A_OPP, DEFECT)),
        "state_opp_last_defect": fmt(last_round_of(HISTORY_A_OPP, DEFECT)),
        "state_own_payoff": fmt(own_a),
        "state_opp_payoff": fmt(opp_a),
        "state_opp_coop_rate": fmt(percent(cooperation_rate(HISTORY_B_OPP))),
        "transition_opp_payoff": fmt(transition_payoff(Fraction("9.8"), DEFECT, COOPERATE)),
        "transition_opp_coop_rate": fmt(percent(transition_cooperation_rate(3, 4, COOPERATE))),
        "terminal_bonus_boundary": fmt(bonus_eligible(3, 5)),
        "terminal_opp_bonus": fmt(bonus_eligible(count(HISTORY_C_OPP, COOPERATE), len(HISTORY_C_OPP))),
        "terminal_own_final": fmt(final_b[0]),
        "terminal_opp_final": fmt(final_b[1]),
        "terminal_own_final_six": fmt(final_a[0]),
        "expected_coop_coop": fmt(expected_final_payoff((), COOPERATE, (), COOPERATE)),
        "expected_defect3_coop": fmt(expected_final_payoff((D, D, D), COOPERATE, (), COOPERATE)),
    }
