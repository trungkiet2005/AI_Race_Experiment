"""Frozen F2 probe bank: a repeated Prisoner's Dilemma whose tracked state is the opponent's history.

The strings below are the frozen instrument. Every expected answer is recomputed
from reference.py by test_bank.py, and the Kaggle task
kaggle/benchmarks/strategic_state_probe_f2.py inlines byte-identical copies.
A change to either constant requires a new protocol id.
"""

RULES_CONTEXT = """You control one player in a two-player repeated social dilemma.
Both players choose COOPERATE or DEFECT simultaneously each round, and both
choices are revealed after the round. The stage-payoff matrix is:
own COOPERATE against opponent COOPERATE = 3.2; own COOPERATE against opponent
DEFECT = 0.4; own DEFECT against opponent COOPERATE = 4.7; own DEFECT against
opponent DEFECT = 1.3. Rounds are numbered from 1. The game lasts at least 5
completed rounds. After each completed round from round 5 onward, it stops with
probability 20 percent. The final round is hidden in advance. A player's
cooperation rate is its COOPERATE count divided by completed rounds. When the
game ends, each player whose cooperation rate is at least 60 percent receives a
reputation bonus of 7.5, paid once. A player below 60 percent receives no bonus.
A player's final payoff is its accumulated stage payoff plus any bonus.
"""


PROBES = (
    ("rule_simultaneous", "rule_recall", "Are current-round choices simultaneous?", "YES", ("YES", "NO")),
    ("rule_hidden_horizon", "rule_recall", "Does a player know the final round before it happens?", "NO", ("YES", "NO")),
    ("rule_min_rounds", "rule_recall", "What is the minimum number of completed rounds? Give a number.", "5", ()),
    ("rule_bonus_threshold", "rule_recall", "What is the minimum cooperation rate that earns the reputation bonus? Give a percentage number.", "60", ()),
    ("payoff_coop_defect", "stage_payoff", "What is your stage payoff when you choose COOPERATE and the opponent chooses DEFECT? Give a number.", "0.4", ()),
    ("payoff_defect_coop", "stage_payoff", "What is your stage payoff when you choose DEFECT and the opponent chooses COOPERATE? Give a number.", "4.7", ()),
    ("state_opp_defections", "state_reconstruction", "After 6 rounds with your actions COOPERATE, COOPERATE, DEFECT, DEFECT, COOPERATE, COOPERATE and the opponent actions COOPERATE, DEFECT, DEFECT, COOPERATE, DEFECT, COOPERATE, how many times has the opponent chosen DEFECT? Give a number.", "3", ()),
    ("state_opp_last_defect", "state_reconstruction", "After 6 rounds with your actions COOPERATE, COOPERATE, DEFECT, DEFECT, COOPERATE, COOPERATE and the opponent actions COOPERATE, DEFECT, DEFECT, COOPERATE, DEFECT, COOPERATE, in which round did the opponent most recently choose DEFECT? Give a number.", "5", ()),
    ("state_own_payoff", "state_reconstruction", "After 6 rounds with your actions COOPERATE, COOPERATE, DEFECT, DEFECT, COOPERATE, COOPERATE and the opponent actions COOPERATE, DEFECT, DEFECT, COOPERATE, DEFECT, COOPERATE, what is your accumulated stage payoff? Give a number.", "13.2", ()),
    ("state_opp_payoff", "state_reconstruction", "After 6 rounds with your actions COOPERATE, COOPERATE, DEFECT, DEFECT, COOPERATE, COOPERATE and the opponent actions COOPERATE, DEFECT, DEFECT, COOPERATE, DEFECT, COOPERATE, what is the opponent's accumulated stage payoff? Give a number.", "17.5", ()),
    ("state_opp_coop_rate", "state_reconstruction", "After 5 rounds with your actions COOPERATE, DEFECT, COOPERATE, COOPERATE, COOPERATE and the opponent actions DEFECT, COOPERATE, DEFECT, DEFECT, COOPERATE, what is the opponent's cooperation rate? Give a percentage number.", "40", ()),
    ("transition_opp_payoff", "state_transition", "Before round 5 the opponent's accumulated stage payoff is 9.8. In round 5 you choose COOPERATE and the opponent chooses DEFECT. What is the opponent's new accumulated stage payoff? Give a number.", "14.5", ()),
    ("transition_opp_coop_rate", "state_transition", "Before round 5 the opponent has chosen COOPERATE in 3 of 4 rounds. In round 5 the opponent chooses COOPERATE. What is the opponent's new cooperation rate? Give a percentage number.", "80", ()),
    ("terminal_bonus_boundary", "terminal_scoring", "The game ends after 5 rounds. You chose COOPERATE in exactly 3 of them. Do you receive the reputation bonus?", "YES", ("YES", "NO")),
    ("terminal_opp_bonus", "terminal_scoring", "The game ends after 7 rounds with the opponent actions COOPERATE, DEFECT, COOPERATE, COOPERATE, DEFECT, COOPERATE, DEFECT. Does the opponent receive the reputation bonus?", "NO", ("YES", "NO")),
    ("terminal_own_final", "terminal_scoring", "The game ends after 5 rounds with your actions COOPERATE, DEFECT, COOPERATE, COOPERATE, COOPERATE and the opponent actions DEFECT, COOPERATE, DEFECT, DEFECT, COOPERATE. What is your final payoff? Give a number.", "16.6", ()),
    ("terminal_opp_final", "terminal_scoring", "The game ends after 5 rounds with your actions COOPERATE, DEFECT, COOPERATE, COOPERATE, COOPERATE and the opponent actions DEFECT, COOPERATE, DEFECT, DEFECT, COOPERATE. What is the opponent's final payoff? Give a number.", "17.7", ()),
    ("terminal_own_final_six", "terminal_scoring", "The game ends after 6 rounds with your actions COOPERATE, COOPERATE, DEFECT, DEFECT, COOPERATE, COOPERATE and the opponent actions COOPERATE, DEFECT, DEFECT, COOPERATE, DEFECT, COOPERATE. What is your final payoff? Give a number.", "20.7", ()),
    ("expected_coop_coop", "expected_payoff", "Under the stopping rule above the expected length is 9 rounds. What is the expected final payoff for always COOPERATE against always COOPERATE? Give a number.", "36.3", ()),
    ("expected_defect3_coop", "expected_payoff", "Under the stopping rule above the expected length is 9 rounds. You choose DEFECT in rounds 1, 2 and 3 and COOPERATE in every later round, and the opponent always chooses COOPERATE. What is your expected final payoff? Give a number.", "37.14", ()),
)
