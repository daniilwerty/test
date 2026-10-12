"""
strategies.py - Step 2: how a player decides to roll again or bank.

Every strategy answers the same question after each scoring roll:
"My turn total is S and I am about to roll n dice. Do I roll again?"

    BreakEven         roll if S < S*_n, otherwise bank
    Adaptive(k)       roll if S < T_n = S*_n x rho^k, otherwise bank,
                      where rho = (G - B) / (G - L)
    FixedThreshold(T) bank as soon as S >= T, whatever n is

B = the player's own banked score, L = the opponent's banked score.

Rules that apply to every strategy (the 'final-turn rule', 'bank at G' and
'entry minimum') live in game.py in the function keeps_rolling(), so that
they are written once and applied the same way to everyone.
"""

import math

from probabilities import break_even_thresholds


def compute_rho(own_banked, opponent_banked, goal):
    """
    rho = (G - B) / (G - L): how far I am from the goal compared with how far
    my opponent is from the goal.

        rho > 1  ->  I am behind, so the adaptive player takes more risk
        rho = 1  ->  level
        rho < 1  ->  I am ahead, so the adaptive player takes less risk

    Guard: if the opponent has already reached the goal (L >= G), the
    denominator is zero or negative and rho has no meaning. That only happens
    on a final turn, where the final-turn rule decides instead. In that case
    this function returns NaN ("not a number") rather than crashing.
    """
    distance_opponent = goal - opponent_banked
    if distance_opponent <= 0:
        return math.nan
    return (goal - own_banked) / distance_opponent


class BreakEven:
    """Roll while the turn total is below the break-even value S*_n."""

    def __init__(self, bonus_mode="off", bonus_aware=True):
        self.name = "break-even"
        self.thresholds = break_even_thresholds(bonus_mode, bonus_aware)

    def threshold(self, n, own_banked, opponent_banked, goal):
        return self.thresholds[n]

    def wants_to_roll(self, turn_total, n, own_banked, opponent_banked, goal):
        return turn_total < self.threshold(n, own_banked, opponent_banked, goal)


class Adaptive:
    """
    Roll while the turn total is below T_n = S*_n x rho^k.

    k = 0 gives rho^0 = 1, so T_n = S*_n: exactly the break-even player.
    Larger k means the player reacts more strongly to being ahead or behind.
    """

    def __init__(self, k, bonus_mode="off", bonus_aware=True):
        if k < 0:
            raise ValueError("k must be 0 or more")
        self.k = k
        self.name = f"adaptive(k={k:g})"
        self.thresholds = break_even_thresholds(bonus_mode, bonus_aware)

    def threshold(self, n, own_banked, opponent_banked, goal):
        rho = compute_rho(own_banked, opponent_banked, goal)
        if math.isnan(rho):
            # Opponent already at the goal: the final-turn rule in game.py is
            # used before this point is reached. Fall back to rho = 1 so the
            # function still returns a sensible number if called directly.
            rho = 1.0
        return self.thresholds[n] * rho ** self.k

    def wants_to_roll(self, turn_total, n, own_banked, opponent_banked, goal):
        return turn_total < self.threshold(n, own_banked, opponent_banked, goal)


class FixedThreshold:
    """Bank as soon as the turn total reaches T, no matter how many dice are left."""

    def __init__(self, T):
        self.T = T
        self.name = f"fixed(T={T})"

    def threshold(self, n, own_banked, opponent_banked, goal):
        return self.T

    def wants_to_roll(self, turn_total, n, own_banked, opponent_banked, goal):
        return turn_total < self.T
