"""
markov.py - Step 5: exact probability of winning a final turn.

On the final turn the chasing player needs a turn total S greater than D.
They keep rolling until S > D (win) or they farkle (lose).

States: (points still needed, n dice about to be rolled).
Writing R = D - S for the points still needed, the chaser wins once R < 0.
    - Farkle            -> absorbing losing state (probability of winning 0)
    - S > D, i.e. R < 0 -> absorbing winning state (probability of winning 1)

From any other state, one roll of n dice moves to a new state with known
probabilities (from the outcome enumeration in scoring.py), so

    P(R, n) = sum over scoring outcomes of  prob(outcome) x P(R - points, n')

where n' = n - dice set aside, or 6 if all dice scored (hot dice).

This recursion always finishes because every scoring roll adds at least 50
points, so R gets smaller every step. Memoisation (lru_cache) stores each
P(R, n) once it has been calculated, so it is reused for every D.
"""

import sys
from collections import Counter
from functools import lru_cache

from scoring import outcome_table

# The recursion can go about 100 levels deep for D = 5000, plus a few extra
# calls per level, so the default limit (1000) is raised to be safe.
sys.setrecursionlimit(20_000)


@lru_cache(maxsize=None)
def roll_transitions(n):
    """
    Group the 6^n outcomes of n dice by (points, dice set aside), with
    the probability of each group. Farkles are left out because they lead
    to a loss and add nothing to the winning probability.

    Example for n = 1: [((100, 1), 1/6), ((50, 1), 1/6)].
    """
    points, used = outcome_table(n)
    total = 6 ** n
    counts = Counter(zip(points.tolist(), used.tolist()))
    return tuple(((p, u), c / total) for (p, u), c in sorted(counts.items()) if p > 0)


def _value_after(points_needed, points, dice_left, bonus_mode):
    """Winning probability after a scoring roll that adds 'points'."""
    new_needed = points_needed - points
    if new_needed < 0:             # S > D reached: winning state
        return 1.0
    if dice_left == 0:             # hot dice: all 6 dice come back
        dice_left = 6
    return win_probability(new_needed, dice_left, bonus_mode)


@lru_cache(maxsize=None)
def win_probability(points_needed, n=6, bonus_mode="off"):
    """
    Exact probability of reaching S > D before a farkle, where
    points_needed = D - S and n dice are about to be rolled.

    Call win_probability(D) for the start of a final turn (S = 0, n = 6).
    """
    if points_needed < 0:
        return 1.0

    # One die with the bonus rule needs special treatment.
    if n == 1 and bonus_mode != "off":
        # a = winning probability after rolling a 1, b = after rolling a 5.
        a = _value_after(points_needed, 100, 0, bonus_mode)
        b = _value_after(points_needed, 50, 0, bonus_mode)
        normal_part = (a + b) / 6
        if bonus_mode == "once":
            # A 6 (prob 1/6) gives one more roll, on which a 6 is a farkle.
            return normal_part + (1 / 6) * normal_part
        if bonus_mode == "chain":
            # A 6 returns us to this same state, so P = (a + b)/6 + (1/6) P.
            # Solving for P gives P = (a + b) / 5.
            return (a + b) / 5
        raise ValueError("bonus_mode must be 'off', 'once' or 'chain'")

    total = 0.0
    for (points, dice_used), probability in roll_transitions(n):
        total += probability * _value_after(points_needed, points, n - dice_used,
                                            bonus_mode)
    return total
