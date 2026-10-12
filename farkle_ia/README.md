# Farkle simulation for an IB Mathematics AI HL Internal Assessment

This project simulates two-player Farkle to answer two questions:

1. Can a player who changes their risk depending on the score (the "adaptive" player) beat a player who always uses the break-even rule?
2. How does a bonus-roll house rule change the length of a game for different play styles? (Main research question.)

It only uses `numpy`, `scipy`, `pandas` and `matplotlib`, plus the Python standard library.

## How to run

```
pip install -r requirements.txt
python -m unittest discover tests     # unit tests
python run_all.py --quick             # small test run, N = 1,000, output in results_quick/
python run_all.py                     # full run, output in results/
```

Every result is saved as a CSV file and every plot as a PNG file in the output folder.

### Run time

Measured on a 4-core cloud machine with Python 3.13 (the code uses 1 core):

| Run | Time |
|---|---|
| `python run_all.py --quick` | about 22 seconds |
| `python run_all.py` (full) | about 5 to 8 minutes (estimate from the quick run; will be replaced with the measured time after the full run) |

One game takes about 0.1 milliseconds. The full run plays about 2.5 million games.

## Reproducibility

`game.py` sets `MASTER_SEED = 2026`. Each experiment gets its own random number generator made from the master seed and a text label, for example `"A coarse bonus=off opponent=break-even k=0.5"`. So:

- running the project twice gives exactly the same numbers;
- different experiments use different, independent random numbers. This is how Experiment C uses new games instead of reusing games from A.

## Files and functions

### `scoring.py`: the scoring rules

| Function | What it does |
|---|---|
| `count_faces(dice)` | Turns a roll such as `[1, 5, 5, 2]` into counts of each face: `(1, 1, 0, 0, 2, 0)`. Only the counts matter for scoring. |
| `best_score_from_counts(counts)` | Tries every way of splitting the roll into scoring groups and keeps the highest total. Returns `(points, dice_used)`. If two readings give the same points, it keeps the one that uses more dice. |
| `score_roll(dice)` | Scores a roll given as a list of faces. `points == 0` means a farkle. |
| `outcome_table(n)` | Lists all 6^n outcomes of n dice and scores each one. The simulation and the probability tables both use this. |

### `probabilities.py`: Step 1

| Function | What it does |
|---|---|
| `roll_statistics(n)` | Exact P(farkle), P(score), E(0) and S*_n = E(0) / P(farkle) for n dice, by averaging over all 6^n outcomes. |
| `one_die_statistics(bonus_mode)` | The same values for one die with the bonus rule `"off"`, `"once"` or `"chain"`. The docstring shows the working, including the geometric series for `"chain"`. |
| `probability_table()` | Table for n = 1 to 6, compared with the hand-calculated values. Anything more than 1% away is flagged. |
| `bonus_table()` | The n = 1 values for the three bonus settings. |
| `break_even_thresholds(bonus_mode, bonus_aware)` | The S*_n values the strategies use. If the bonus rule is on and `bonus_aware` is True, S*_1 is the bonus-adjusted value. |

### `strategies.py`: Step 2

| Item | What it does |
|---|---|
| `compute_rho(B, L, G)` | rho = (G - B) / (G - L). Returns NaN when L >= G, so it never divides by zero. |
| `BreakEven` | Rolls while S < S*_n. |
| `Adaptive(k)` | Rolls while S < S*_n × rho^k. With k = 0 this is exactly `BreakEven`. |
| `FixedThreshold(T)` | Rolls while S < T, whatever n is. |

Each strategy has `wants_to_roll(S, n, B, L, G)` (the decision) and `threshold(n, B, L, G)` (the value S is compared with).

### `game.py`: turns, games and logging (Step 3)

| Item | What it does |
|---|---|
| `GameSettings` | All switchable rules: goal G, entry minimum, bonus mode, bonus awareness, and the rho value above which rho is logged. |
| `make_rng(label)` | Makes the reproducible random number generator for one experiment. |
| `DiceRoller` | Hands out random outcome numbers. It gets 100,000 random numbers from numpy at a time, for speed. |
| `bonus_roll_allowed(mode, used)` | Says whether a 6 on a single die gives a bonus roll. |
| `roll_dice(n, ...)` | One roll of n dice, including the bonus rule. Returns `(points, dice_used)` and reports every bonus trigger. |
| `keeps_rolling(...)` | The decision after each scoring roll. It checks, in this order: final-turn rule, bank at G, entry minimum, then the strategy. |
| `play_turn(...)` | One whole turn: rolls, farkles, hot dice and banking. Returns the points banked. |
| `play_game(...)` | A full game, including the final round. Returns the game record and adds bonus triggers and large rho values to the logs. |

Game record fields: `winner`, `starter`, `turns`, `rounds`, `trigger_round`, `triggered_by`, `score_A`, `score_B`, and the two strategy names.

Bonus trigger log fields: `game_id`, `player`, `strategy`, `own_banked`, `opponent_banked`, `turn_total`, `rho`, `final_turn`, `deficit`, and `bonus_number` (1 for the first bonus roll, 2 for the second in a chain, and so on).

### `markov.py`: Step 5

| Function | What it does |
|---|---|
| `roll_transitions(n)` | Groups the 6^n outcomes by (points, dice set aside) with their probabilities. Farkles are left out because they lead to a loss. |
| `win_probability(R, n, bonus_mode)` | Exact probability that the chaser reaches S > D before a farkle. Here R = D - S is the number of points still needed. It uses recursion with memoisation. It always finishes because every scoring roll adds at least 50 points. |

With the `"chain"` bonus, a 6 on one die leads back to the same state. This gives the equation P = (a + b)/6 + P/6, so P = (a + b)/5, where a and b are the winning probabilities after rolling a 1 and after rolling a 5.

### `experiments.py`: Step 4 and the Step 5 check

| Function | What it does |
|---|---|
| `play_match(A, B, N, settings, label)` | Plays N games and alternates who starts (A starts the even-numbered games). Returns a game table, a bonus-trigger table and a large-rho table. |
| `win_rate_summary(games)` | Wins, N, p, SE = sqrt(p(1 - p)/N), and the 95% CI p ± 1.96 SE. |
| `k_sweep(...)` | Plays adaptive(k) against one opponent for every k in a list. |
| `fit_quadratic(k, p)` | Least-squares fit of p(k) = ak² + bk + c with `numpy.polyfit`. Returns a, b, c, R² and k* = -b/(2a). |
| `step1_probabilities` | Saves the Step 1 tables. |
| `experiment_A` | Coarse grid k = 0 to 3 in steps of 0.25, then a fine grid in steps of 0.05 from best k - 0.25 to best k + 0.25. Bonus off and on. |
| `experiment_B` | Quadratic fit to the fine-grid points, plus plots. |
| `experiment_C` | New games and the chi-squared test of independence. |
| `experiment_D` | Coarse grid against fixed(T), for T = 300, 500, 1000 and 2000. |
| `experiment_E` | Game length: mirror matches and the full matrix, the plots, and the rho histogram. |
| `experiment_markov` | Exact final-turn curve for D = 50 to 5,000, checked against simulated final turns. |

### `run_all.py`

Runs everything in order, prints every table, and records the time each step takes.

### `tests/test_scoring.py`

- 23 hand-checked rolls: farkles, singles, three/four/five/six of a kind, the straight, three pairs, four of a kind + pair, two triplets, and cases where the highest-scoring reading matters.
- A check that adaptive(k = 0) makes the same decision as break-even. It covers every n, turn totals 0 to 20,000, five score situations, and all three bonus modes.

## Assumptions

### Rules and scoring

1. Two players, six standard fair dice, G = 10,000, entry minimum 0. The entry minimum is set with `GameSettings.entry_minimum`.
2. A roll is scored using its highest-scoring reading. For example, four 1s score 1100 (three 1s + a single 1), not 1000, and 1-1-1-1-5-5 scores 1500 (four of a kind + a pair), not 1200.
3. "Four of a kind + a pair" and "three pairs" need two or three different faces. Six of the same face always scores 3000, which is higher anyway.
4. Six-dice combinations (straight, three pairs, four + pair, two triplets) only count when all six dice are rolled together.
5. The player always sets aside every die that the highest-scoring reading uses. Each roll is scored on its own. Dice set aside earlier never combine with later dice.
6. Hot dice: when every remaining die scores, the player keeps S and gets all 6 dice back. The strategy then decides with n = 6.
7. The bonus-roll rule only applies when exactly one die is rolled. The bonus roll is forced (the player cannot bank instead). If the bonus roll shows a 1 or a 5, that die scores, which is hot dice. If it shows a 2, 3 or 4, it is a farkle. A 6 on a bonus roll is a farkle under `"once"` and another bonus roll under `"chain"`. In the experiments, "bonus on" means `"chain"`, the default.
8. Each 6 that gives a bonus roll is logged as one trigger. So a chain of two 6s is two triggers.

### Strategies

9. **Bank at G** (as you confirmed): in a normal turn, every strategy banks as soon as B + S >= G, whatever its threshold says.
10. **Final-turn rule:** when the opponent has reached G, the chaser rolls while S <= D, where D = opponent's score - own banked score. They bank as soon as S > D. A tie is a loss. This overrides the strategy.
11. After a player banks to G or above, the other player gets one final turn. The game then ends, whatever happens in that turn.
12. Bonus awareness (`bonus_aware = True` by default): when the bonus rule is on, the strategies use the bonus-adjusted S*_1. This is 50 for `"chain"` and 47.73 for `"once"`, instead of 37.5.
13. A fixed-threshold player uses the same T whether or not the bonus rule is on.
14. rho is undefined when L >= G (division by zero). That only happens on a final turn, where the final-turn rule decides. `compute_rho` returns NaN there, and `Adaptive.threshold` uses rho = 1 if it is ever called in that case.
15. "Very large rho" means rho > 10 (setting `large_rho_limit`). It is checked once at the start of each turn of an adaptive player, and logged.

### Logging and game length

16. **Rounds** (as you defined it): rounds = turns taken by the starting player = ceil(total turns / 2). The final turn counts as a turn. **Trigger round** = ceil(turn number when G was first reached / 2).

### Experiments

17. In every match, adaptive(k) is player A and the opponent is player B. A starts the even-numbered games and B the odd-numbered ones, so each starts exactly N/2 games. The win rate p is A's win rate.
18. **A:** the fine grid runs in steps of 0.05 from best coarse k - 0.25 to best coarse k + 0.25, never below 0. It uses new games, not the coarse-grid games.
19. **B:** "points near the peak" means the fine-grid points. If the fit gives a >= 0 (a curve with no maximum), k* is undefined. In that case the best observed fine-grid k is used for C and E, and the table says so. k* is rounded to 3 decimal places before it is used.
20. **C:** run for bonus off and on, each with its own k*. The 2x2 table has rows (adaptive(k*) vs break-even, break-even vs break-even) and columns (player A wins, player A loses). The chi-squared test uses no Yates continuity correction, the same as a GDC. Degrees of freedom = 1.
21. **D:** N = 10,000 per k, coarse grid only, bonus off and on. k* for each opponent comes from a quadratic fit to the best coarse k and its neighbours within ±0.5 (up to 5 points). The best observed k is also reported, because a 5-point fit on a noisy curve can be unreliable.
22. **E:** the main design is mirror matches with N = 20,000. The secondary design is a full matrix with N = 5,000 per pairing. Each unordered pair is played once, because with alternating starts "X vs Y" and "Y vs X" are the same experiment. adaptive(k*) uses the k* for the matching bonus setting. Game length is measured in turns, rounds and trigger round.
23. **E, rho histogram:** the triggers come from the mirror matches. On a final turn rho is undefined (see 14). So final-turn triggers are shown in a separate panel by their deficit D instead of rho. Normal-turn triggers with rho > 5 are counted in the plot title but not drawn, so the main shape stays readable. All values are in `E_bonus_triggers.csv`.
24. **Step 5 check:** simulated final turns start with own banked score 0 and opponent score D, so the chaser needs S > D. N = 100,000 per D in the full run.
25. The quick run uses N = 1,000 for everything, including the Markov check.

## Summary of key results

**Quick test run only (N = 1,000). These are not final results.** With N = 1,000 the 95% CI is about ±0.031, so the win rates in this table cannot be told apart. This table will be replaced after the full run.

| Result | Bonus off | Bonus on (chain) |
|---|---|---|
| S*_1 | 37.5 | 50.0 |
| S*_2 to S*_6 | 112.5, 312.5, 912.25, 2930.42, 18368.06 | same |
| All S*_n within 1% of hand values | yes (largest difference 0.03%) | — |
| k* (quadratic fit) | no maximum in quick run (a > 0) | no maximum in quick run (a > 0) |
| Chi-squared p-value (C) | not significant (quick run) | not significant (quick run) |
| Mean game length, break-even mirror | 33.5 turns | 33.3 turns |
| Mean game length, fixed T = 2000 mirror | 70.3 turns | 63.1 turns |
| P(win final turn), D = 1000 (exact) | 0.2830 | 0.3000 |
| Markov check | 5 of 6 within 95% CI at N = 1,000 | |

Full quick-run output: `results_quick/quick_run_output.txt`.
