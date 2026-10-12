"""
scoring.py - the Farkle scoring rules.

This module answers one question: "If I roll these dice, how many points do
they score, and how many of the dice score?"

The rules (from the IA brief):
    single 1                 = 100
    single 5                 = 50
    three 1s                 = 1000
    three of a kind (2 to 6) = 100 x face value  (e.g. three 4s = 400)
    four of a kind           = 1000
    five of a kind           = 2000
    six of a kind            = 3000
    straight 1-2-3-4-5-6     = 1500
    three pairs              = 1500
    four of a kind + a pair  = 1500
    two triplets             = 2500

A roll can often be read in more than one way. For example, four 1s can be
read as "four of a kind" (1000) or as "three 1s + a single 1" (1000 + 100).
The brief says to always use the highest-scoring reading, so the code tries
every possible way of splitting the roll into scoring groups and keeps the
best one.

The player's policy is to set aside every scoring die. So the dice that are
"used" by the best reading are the dice that get set aside, and the rest are
rolled again.
"""

import itertools
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Small helper: turn a roll into "how many of each face"
# ---------------------------------------------------------------------------

def count_faces(dice):
    """
    Turn a list of dice, e.g. [1, 5, 5, 2], into a tuple of six counts:
    (number of 1s, number of 2s, ..., number of 6s).
    For [1, 5, 5, 2] this gives (1, 1, 0, 0, 2, 0).

    Only the counts matter for scoring, not the order of the dice.
    """
    counts = [0, 0, 0, 0, 0, 0]
    for face in dice:
        counts[face - 1] += 1
    return tuple(counts)


# ---------------------------------------------------------------------------
# The list of scoring groups that can be taken out of a roll one at a time
# ---------------------------------------------------------------------------

def _single_groups():
    """
    Build the list of scoring groups that can be removed from any roll.
    Each group is (counts_needed, points). For example "three 4s" is
    ((0, 0, 0, 3, 0, 0), 400).
    """
    groups = []

    # Single 1 and single 5.
    groups.append(((1, 0, 0, 0, 0, 0), 100))
    groups.append(((0, 0, 0, 0, 1, 0), 50))

    # Three, four, five and six of a kind, for every face.
    for face in range(1, 7):
        if face == 1:
            three_points = 1000
        else:
            three_points = 100 * face
        for how_many, points in ((3, three_points), (4, 1000), (5, 2000), (6, 3000)):
            needed = [0, 0, 0, 0, 0, 0]
            needed[face - 1] = how_many
            groups.append((tuple(needed), points))

    return groups


SCORING_GROUPS = _single_groups()


def _six_dice_combination_points(counts):
    """
    The four combinations that need all six dice at once:
    straight, three pairs, four of a kind + a pair, and two triplets.

    Returns the points for the best one that fits, or 0 if none fit.
    'counts' must describe exactly six dice.
    """
    # Sort the non-zero counts so that, for example, (2, 0, 2, 0, 0, 2)
    # becomes [2, 2, 2], which is easy to compare against a pattern.
    pattern = sorted(c for c in counts if c > 0)

    best = 0
    if pattern == [1, 1, 1, 1, 1, 1]:   # straight 1-6
        best = max(best, 1500)
    if pattern == [2, 2, 2]:            # three pairs
        best = max(best, 1500)
    if pattern == [2, 4]:               # four of a kind + a pair
        best = max(best, 1500)
    if pattern == [3, 3]:               # two triplets
        best = max(best, 2500)
    return best


# ---------------------------------------------------------------------------
# The main scoring function
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def best_score_from_counts(counts):
    """
    Find the highest-scoring reading of a roll, given its face counts.

    Returns a pair (points, dice_used):
        points    - the maximum score the roll can make
        dice_used - how many dice that best reading sets aside

    Method: try removing each scoring group that fits, score what is left
    in the same way (recursion), and keep the best total. If two readings
    give the same points, the one that uses more dice is kept, because the
    policy is to set aside every scoring die.

    The result is remembered (lru_cache) so each pattern is only worked out
    once.
    """
    dice_left = sum(counts)
    best = (0, 0)   # (points, dice_used); scoring nothing is always possible

    # Six-dice combinations can only be used when all six dice are present.
    if dice_left == 6:
        points = _six_dice_combination_points(counts)
        if points > 0:
            best = max(best, (points, 6))

    # Try taking out each ordinary scoring group.
    for needed, points in SCORING_GROUPS:
        fits = all(have >= need for have, need in zip(counts, needed))
        if not fits:
            continue
        remaining = tuple(have - need for have, need in zip(counts, needed))
        rest_points, rest_used = best_score_from_counts(remaining)
        candidate = (points + rest_points, sum(needed) + rest_used)
        best = max(best, candidate)   # compares points first, then dice used

    return best


def score_roll(dice):
    """
    Score a roll given as a list of faces, e.g. score_roll([1, 1, 5, 2, 3, 4]).

    Returns (points, dice_used). A roll with points == 0 is a farkle.
    """
    if not 1 <= len(dice) <= 6:
        raise ValueError("A roll must have between 1 and 6 dice.")
    if any(face not in (1, 2, 3, 4, 5, 6) for face in dice):
        raise ValueError("Every die must show a face from 1 to 6.")
    return best_score_from_counts(count_faces(dice))


# ---------------------------------------------------------------------------
# Lookup tables of every possible outcome (used by the simulation and Step 1)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def outcome_table(n):
    """
    List every one of the 6^n equally likely outcomes of rolling n dice and
    score each one.

    Returns two numpy arrays of length 6^n:
        points[i]    - score of outcome number i
        dice_used[i] - how many dice outcome number i sets aside

    Outcome number i is the i-th roll in the order produced by
    itertools.product, i.e. (1,1,...,1), (1,1,...,2), ..., (6,6,...,6).
    So for one die, outcome 0 is a 1 and outcome 5 is a 6.

    The simulation picks a random outcome number and looks its score up
    here, which is much faster than scoring each roll from scratch.
    """
    points = []
    dice_used = []
    for roll in itertools.product(range(1, 7), repeat=n):
        p, u = score_roll(roll)
        points.append(p)
        dice_used.append(u)
    return np.array(points, dtype=np.int64), np.array(dice_used, dtype=np.int64)
