"""
game.py - simulating turns and whole games of two-player Farkle.

The main functions are:
    roll_dice()     one roll of n dice, including the bonus rule
    play_turn()     one player's whole turn
    play_game()     a full game between two strategies

Logging (Step 3): play_game() returns a record of the game (winner, who
started, number of turns and rounds, final scores, trigger round), and adds
every bonus-rule trigger and every very large rho value to lists that the
caller passes in.
"""

import math
import zlib
from dataclasses import dataclass

import numpy as np

from scoring import outcome_table
from strategies import Adaptive, compute_rho


# One fixed master seed makes every result reproducible.
MASTER_SEED = 2026


@dataclass
class GameSettings:
    """All switchable rules in one place."""
    goal: int = 10_000              # target score G
    entry_minimum: int = 0          # minimum first bank (0 = no entry minimum)
    bonus_mode: str = "chain"       # "off", "once" or "chain"
    bonus_aware: bool = True        # do players use the bonus-adjusted S*_1?
    large_rho_limit: float = 10.0   # rho above this value is logged


def make_rng(label):
    """
    Make a random number generator for one experiment.

    The seed is built from the master seed and the experiment's label, e.g.
    "A coarse bonus=off k=0.5". So each experiment always gets the same
    random numbers (reproducible), but different experiments get different,
    independent random numbers (so Experiment C really uses new games).
    """
    label_number = zlib.crc32(label.encode("utf-8"))
    return np.random.default_rng([MASTER_SEED, label_number])


class DiceRoller:
    """
    Produces random dice outcomes.

    Asking numpy for one random number at a time is slow, so this class asks
    for 100,000 at once and hands them out one by one. Each number u is
    uniform on [0, 1), and int(u x m) is then a uniformly random whole number
    from 0 to m - 1. With m = 6^n this picks one of the 6^n equally likely
    outcomes of n dice.
    """

    def __init__(self, rng, batch_size=100_000):
        self.rng = rng
        self.batch_size = batch_size
        self.batch = rng.random(batch_size)
        self.position = 0

    def random_index(self, how_many_outcomes):
        if self.position == self.batch_size:
            self.batch = self.rng.random(self.batch_size)
            self.position = 0
        u = self.batch[self.position]
        self.position += 1
        return int(u * how_many_outcomes)


# Pre-computed scores for every outcome of 1..6 dice (from scoring.py).
OUTCOME_POINTS = {n: outcome_table(n)[0].tolist() for n in range(1, 7)}
OUTCOME_DICE_USED = {n: outcome_table(n)[1].tolist() for n in range(1, 7)}


def bonus_roll_allowed(bonus_mode, bonus_rolls_so_far):
    """
    Can a 6 on a single die be turned into a bonus roll?
        "off":   never
        "once":  only if no bonus roll has been used yet for this die
        "chain": always (a 6 on a bonus roll gives another bonus roll)
    """
    if bonus_mode == "off":
        return False
    if bonus_mode == "once":
        return bonus_rolls_so_far == 0
    if bonus_mode == "chain":
        return True
    raise ValueError("bonus_mode must be 'off', 'once' or 'chain'")


def roll_dice(n, roller, settings, on_bonus_trigger=None):
    """
    Roll n dice once and return (points, dice_used).

    points == 0 means a farkle. The player always sets aside every scoring
    die, so dice_used is the number of dice set aside.

    Bonus rule: if a single die is rolled and shows a 6, it is not a farkle;
    the die is rolled again (as allowed by settings.bonus_mode). Each time
    this happens, on_bonus_trigger(bonus_number) is called so it can be logged.
    """
    outcome = roller.random_index(6 ** n)

    if n == 1 and settings.bonus_mode != "off":
        face = outcome + 1          # for one die, outcome 0..5 means face 1..6
        bonus_rolls = 0
        while face == 6 and bonus_roll_allowed(settings.bonus_mode, bonus_rolls):
            bonus_rolls += 1
            if on_bonus_trigger is not None:
                on_bonus_trigger(bonus_rolls)
            face = roller.random_index(6) + 1
        # After the bonus rolls (if any), score the face that is showing.
        # A 6 here means the bonus roll is not allowed, so it is a farkle.
        if face == 1:
            return 100, 1
        if face == 5:
            return 50, 1
        return 0, 0

    return OUTCOME_POINTS[n][outcome], OUTCOME_DICE_USED[n][outcome]


def keeps_rolling(strategy, turn_total, n, own_banked, opponent_banked,
                  final_turn, settings):
    """
    Decide whether to roll again. The shared rules are checked first, in
    this order, and the strategy only decides if none of them applies.

    1. Final-turn rule: if the opponent has already reached G, keep rolling
       until S > D, where D = opponent's score - own banked score. A tie is
       a loss, so S must be strictly greater than D.
    2. Bank at G: if banking now would reach the goal, bank.
    3. Entry minimum: a player with nothing banked must reach the entry
       minimum before banking (no effect when the minimum is 0).
    4. Otherwise ask the strategy.
    """
    if final_turn:
        deficit = opponent_banked - own_banked
        return turn_total <= deficit

    if own_banked + turn_total >= settings.goal:
        return False

    if own_banked == 0 and turn_total < settings.entry_minimum:
        return True

    return strategy.wants_to_roll(turn_total, n, own_banked, opponent_banked,
                                  settings.goal)


def play_turn(strategy, own_banked, opponent_banked, final_turn, settings,
              roller, bonus_log=None, log_context=None):
    """
    Play one whole turn and return the points banked (0 after a farkle).

    The turn starts with all 6 dice. After each roll:
      - no scoring dice  -> farkle, the turn total is lost, return 0
      - otherwise add the points to the turn total S and set the scoring
        dice aside; if every die has now scored (hot dice) the player gets
        all 6 dice back
      - then keeps_rolling() decides whether to roll again or bank
    """
    turn_total = 0
    dice_to_roll = 6

    def record_bonus(bonus_number):
        # Called by roll_dice() each time the bonus rule is used.
        if bonus_log is None:
            return
        entry = dict(log_context or {})
        entry.update({
            "strategy": strategy.name,
            "own_banked": own_banked,
            "opponent_banked": opponent_banked,
            "turn_total": turn_total,
            "rho": compute_rho(own_banked, opponent_banked, settings.goal),
            "final_turn": final_turn,
            "deficit": opponent_banked - own_banked if final_turn else math.nan,
            "bonus_number": bonus_number,   # 1 = first bonus roll, 2 = second in a chain...
        })
        bonus_log.append(entry)

    while True:
        points, dice_used = roll_dice(dice_to_roll, roller, settings, record_bonus)

        if points == 0:              # farkle
            return 0

        turn_total += points
        dice_to_roll -= dice_used
        if dice_to_roll == 0:        # hot dice
            dice_to_roll = 6

        if not keeps_rolling(strategy, turn_total, dice_to_roll, own_banked,
                             opponent_banked, final_turn, settings):
            return turn_total


def play_game(strategy_a, strategy_b, a_starts, settings, roller,
              bonus_log=None, large_rho_log=None, game_id=0):
    """
    Play one full game between player A and player B.

    Players take turns. As soon as a player banks to G or above, the other
    player gets exactly one final turn and must beat that score strictly.

    Returns a dictionary describing the game:
        winner         "A" or "B"
        starter        "A" or "B"
        turns          total number of turns played (including the final turn)
        rounds         turns taken by the starting player = ceil(turns / 2)
        trigger_round  the round in which a player first reached G
        score_A, score_B, triggered_by
    """
    strategies = {"A": strategy_a, "B": strategy_b}
    scores = {"A": 0, "B": 0}
    other = {"A": "B", "B": "A"}

    current = "A" if a_starts else "B"
    starter = current
    turns = 0

    while True:
        turns += 1
        opponent = other[current]

        # Log very large rho values for adaptive players (once per turn).
        if large_rho_log is not None and isinstance(strategies[current], Adaptive):
            rho = compute_rho(scores[current], scores[opponent], settings.goal)
            if rho > settings.large_rho_limit:
                large_rho_log.append({
                    "game_id": game_id, "player": current,
                    "strategy": strategies[current].name,
                    "own_banked": scores[current],
                    "opponent_banked": scores[opponent], "rho": rho,
                })

        context = {"game_id": game_id, "player": current}
        scores[current] += play_turn(strategies[current], scores[current],
                                     scores[opponent], False, settings, roller,
                                     bonus_log, context)

        if scores[current] >= settings.goal:
            # The current player has reached the goal: final turn for the other.
            trigger_turn = turns
            triggered_by = current
            turns += 1
            context = {"game_id": game_id, "player": opponent}
            scores[opponent] += play_turn(strategies[opponent], scores[opponent],
                                          scores[current], True, settings,
                                          roller, bonus_log, context)
            # The chaser wins only by beating the score strictly.
            if scores[opponent] > scores[current]:
                winner = opponent
            else:
                winner = current
            break

        current = opponent

    return {
        "game_id": game_id,
        "winner": winner,
        "starter": starter,
        "turns": turns,
        "rounds": math.ceil(turns / 2),
        "trigger_round": math.ceil(trigger_turn / 2),
        "triggered_by": triggered_by,
        "score_A": scores["A"],
        "score_B": scores["B"],
        "strategy_A": strategy_a.name,
        "strategy_B": strategy_b.name,
    }
