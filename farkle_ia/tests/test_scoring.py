"""
Unit tests for the scoring function and for the strategies.

Run from the farkle_ia folder with:
    python -m unittest discover tests
"""

import os
import sys
import unittest

# Make the modules in the parent folder importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import score_roll                      # noqa: E402
from strategies import Adaptive, BreakEven          # noqa: E402


class TestScoring(unittest.TestCase):
    """Each roll below was checked by hand. Format: (dice, points, dice used)."""

    HAND_CHECKED = [
        # Farkles
        ([2, 3, 4, 6, 6, 2], 0, 0),     # no 1s, no 5s, no triple, no special
        ([6], 0, 0),
        ([2, 3], 0, 0),
        # Singles
        ([1], 100, 1),
        ([5], 50, 1),
        ([1, 5, 2, 3], 150, 2),         # 100 + 50, the 2 and 3 do not score
        # Three of a kind
        ([1, 1, 1], 1000, 3),
        ([2, 2, 2], 200, 3),
        ([6, 6, 6, 1, 5], 750, 5),      # 600 + 100 + 50
        ([5, 5, 5, 5], 1000, 4),        # four of a kind beats 500 + 50
        # Four, five, six of a kind
        ([1, 1, 1, 1], 1100, 4),        # three 1s + single 1 (1100) beats four of a kind (1000)
        ([3, 3, 3, 3, 3], 2000, 5),
        ([4, 4, 4, 4, 4, 4], 3000, 6),
        ([1, 1, 1, 1, 1], 2000, 5),     # five of a kind (2000) beats 1000 + 100 + 100
        # Six-dice combinations
        ([1, 2, 3, 4, 5, 6], 1500, 6),  # straight
        ([6, 3, 5, 1, 4, 2], 1500, 6),  # straight in any order
        ([2, 2, 3, 3, 4, 4], 1500, 6),  # three pairs
        ([2, 2, 2, 2, 3, 3], 1500, 6),  # four of a kind + a pair
        ([1, 1, 1, 1, 5, 5], 1500, 6),  # four + pair (1500) beats 1100 + 100
        ([2, 2, 2, 3, 3, 3], 2500, 6),  # two triplets
        ([1, 1, 1, 6, 6, 6], 2500, 6),  # two triplets (2500) beats 1000 + 600
        # Mixed
        ([2, 2, 2, 1, 4, 6], 300, 4),   # 200 + 100
        ([3, 3, 4, 4, 6, 1], 100, 1),   # only two pairs, so just the single 1
    ]

    def test_hand_checked_rolls(self):
        for dice, points, used in self.HAND_CHECKED:
            with self.subTest(dice=dice):
                self.assertEqual(score_roll(dice), (points, used))

    def test_order_does_not_matter(self):
        self.assertEqual(score_roll([5, 1, 3]), score_roll([3, 5, 1]))

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            score_roll([7])
        with self.assertRaises(ValueError):
            score_roll([1, 1, 1, 1, 1, 1, 1])


class TestStrategies(unittest.TestCase):

    def test_adaptive_k0_equals_break_even(self):
        """adaptive(k = 0) must make exactly the same decision as break-even."""
        goal = 10_000
        for bonus_mode in ("off", "once", "chain"):
            adaptive = Adaptive(0, bonus_mode)
            break_even = BreakEven(bonus_mode)
            for n in range(1, 7):
                for turn_total in range(0, 20_001, 50):
                    for own, opponent in [(0, 0), (0, 9_950), (9_900, 0),
                                          (4_000, 7_000), (7_000, 4_000)]:
                        self.assertEqual(
                            adaptive.wants_to_roll(turn_total, n, own, opponent, goal),
                            break_even.wants_to_roll(turn_total, n, own, opponent, goal))

    def test_adaptive_takes_more_risk_when_behind(self):
        adaptive = Adaptive(1)
        behind = adaptive.threshold(3, 2_000, 8_000, 10_000)
        level = adaptive.threshold(3, 5_000, 5_000, 10_000)
        ahead = adaptive.threshold(3, 8_000, 2_000, 10_000)
        self.assertGreater(behind, level)
        self.assertGreater(level, ahead)


if __name__ == "__main__":
    unittest.main()
