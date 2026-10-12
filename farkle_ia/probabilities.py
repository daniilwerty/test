"""
probabilities.py - Step 1: exact probability tables.

For n = 1 to 6 dice we list all 6^n equally likely outcomes and work out:

    P(farkle) - probability that the roll scores nothing
    P(score)  - probability that the roll scores something (= 1 - P(farkle))
    E(0)      - expected score of ONE roll starting from a turn total of 0,
                counting a farkle as 0 points
    S*_n      - the break-even turn total, S*_n = E(0) / P(farkle)

Why S*_n is the break-even point: with turn total S and n dice, rolling once
more gains E(0) on average but loses the whole S with probability P(farkle).
So the expected change is E(0) - S x P(farkle). This is positive while
S < E(0) / P(farkle) and negative after it. (This looks one roll ahead only.)

The bonus rule only changes the n = 1 row, because it only applies when a
single die is rolled. Those values are worked out separately below.
"""

import pandas as pd

from scoring import outcome_table


# The values from the hand calculations in the IA, used as a check.
HAND_CALCULATED_BREAK_EVEN = {1: 37.5, 2: 112.5, 3: 312.5, 4: 912, 5: 2930, 6: 18368}

# Allowed settings for the bonus rule.
BONUS_MODES = ("off", "once", "chain")


def roll_statistics(n):
    """
    Exact statistics for one roll of n dice, with no bonus rule.

    Every outcome has probability 1 / 6^n, so a probability is
    (number of outcomes) / 6^n and an expected value is an average
    over all outcomes.
    """
    points, _ = outcome_table(n)
    total_outcomes = 6 ** n
    farkle_outcomes = int((points == 0).sum())

    p_farkle = farkle_outcomes / total_outcomes
    p_score = 1 - p_farkle
    expected_score = points.sum() / total_outcomes   # farkles add 0
    break_even = expected_score / p_farkle

    return {
        "n": n,
        "outcomes": total_outcomes,
        "farkle_outcomes": farkle_outcomes,
        "P_farkle": p_farkle,
        "P_score": p_score,
        "E0": expected_score,
        "S_star": break_even,
    }


def one_die_statistics(bonus_mode):
    """
    Exact statistics for rolling a single die, with the bonus rule set to
    "off", "once" or "chain".

    One die: a 1 scores 100, a 5 scores 50, anything else is a farkle.
    With the bonus rule, a 6 is not a farkle: the die is rolled again.

    "off":   P(farkle) = 4/6 (faces 2, 3, 4, 6), E(0) = (100 + 50)/6 = 25.

    "once":  a 6 gives one extra roll, and on that extra roll a 6 IS a farkle.
             P(score)  = 2/6 + (1/6)(2/6)
             E(0)      = 25  + (1/6)(25)

    "chain": every 6 gives another roll. The chance of reaching the k-th
             extra roll is (1/6)^k, so the totals are geometric series:
             P(score) = (2/6)(1 + 1/6 + 1/6^2 + ...) = (2/6) / (1 - 1/6) = 2/5
             E(0)     = 25  (1 + 1/6 + 1/6^2 + ...) = 25 / (5/6)     = 30
    """
    p_score_plain = 2 / 6      # a 1 or a 5
    e_plain = (100 + 50) / 6   # = 25
    p_six = 1 / 6

    if bonus_mode == "off":
        p_score = p_score_plain
        expected_score = e_plain
    elif bonus_mode == "once":
        p_score = p_score_plain + p_six * p_score_plain
        expected_score = e_plain + p_six * e_plain
    elif bonus_mode == "chain":
        p_score = p_score_plain / (1 - p_six)
        expected_score = e_plain / (1 - p_six)
    else:
        raise ValueError(f"bonus_mode must be one of {BONUS_MODES}")

    p_farkle = 1 - p_score
    return {
        "bonus_mode": bonus_mode,
        "P_farkle": p_farkle,
        "P_score": p_score,
        "E0": expected_score,
        "S_star": expected_score / p_farkle,
    }


def probability_table():
    """
    Build the Step 1 table for n = 1..6 (bonus rule off) and compare S*_n with
    the hand-calculated values. Any value more than 1% away is flagged.
    """
    rows = []
    for n in range(1, 7):
        row = roll_statistics(n)
        hand = HAND_CALCULATED_BREAK_EVEN[n]
        row["S_star_hand"] = hand
        row["percent_difference"] = 100 * (row["S_star"] - hand) / hand
        row["flag_over_1_percent"] = abs(row["percent_difference"]) > 1
        rows.append(row)
    return pd.DataFrame(rows)


def bonus_table():
    """The n = 1 values for each bonus-rule setting, side by side."""
    return pd.DataFrame([one_die_statistics(mode) for mode in BONUS_MODES])


def break_even_thresholds(bonus_mode="off", bonus_aware=True):
    """
    The S*_n values that the strategies use, as a dictionary {n: S*_n}.

    If the bonus rule is on AND players know about it (bonus_aware=True),
    S*_1 is replaced by the bonus-adjusted value for that bonus mode.
    Otherwise all values are the plain ones from the bonus-off table.
    """
    thresholds = {n: roll_statistics(n)["S_star"] for n in range(1, 7)}
    if bonus_mode != "off" and bonus_aware:
        thresholds[1] = one_die_statistics(bonus_mode)["S_star"]
    return thresholds
