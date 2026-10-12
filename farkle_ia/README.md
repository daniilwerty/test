# Farkle simulation for an IB Mathematics AI HL Internal Assessment

This project simulates two-player Farkle to answer two questions:

1. How does a bonus-roll house rule change the length of a game for different play styles? (Main research question.)
2. Can a player who changes their risk depending on the score (the "adaptive" player) beat a player who always uses the break-even rule?

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
| `python run_all.py --quick` | about 20 seconds |
| `python run_all.py` (full) | about 205 seconds (3.5 minutes), measured twice: 205.0 s and 203.6 s |

One game takes about 0.1 milliseconds. Because the k* rule gave case 3, the fine grid was not run, so the full run played about 1.9 million games plus 600,000 simulated final turns. If case 2 happens (for example with a different seed or different rules), the fine grid adds up to about 1 minute.

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
| `fit_quadratic(k, p)` | Least-squares fit of p(k) = ak² + bk + c with `numpy.polyfit`. Returns a, b, c, R² and the vertex -b/(2a). |
| `choose_k_star(k, p)` | The decision rule for k*, which was fixed before the full run (see "Decision rule for k*" below). Returns the fit, which case happened, and k*. |
| `k_to_test(fit, coarse)` | Picks the k that Experiment C uses for the adaptive player. This is k* when k* > 0, and otherwise the best observed coarse k (see the decision rule, step 5). |
| `step1_probabilities` | Saves the Step 1 tables. |
| `experiment_A_coarse` | Coarse grid k = 0 to 3 in steps of 0.25, bonus off and on. |
| `experiment_B` | Applies the decision rule to the whole coarse grid. |
| `experiment_A_fine` | Fine grid in steps of 0.05 around k*, only where case 2 happened. |
| `plot_B` | Win rate against k with 95% error bars and the fitted curve on [0, 3]. |
| `experiment_C` | New games and the chi-squared test of independence. |
| `experiment_D` | Coarse grid against fixed(T), for T = 300, 500, 1000 and 2000. The summary has a k = 0 row and a k* row for each opponent. |
| `experiment_E` | Game length: mirror matches and the full matrix, the plots, and the rho histogram. The adaptive player uses `E_ADAPTIVE_K` = 0.5 in both bonus settings. |
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
18. **Decision rule for k\*** (your rule, fixed before the full run):
    1. Fit p(k) = ak² + bk + c to the whole coarse grid (k = 0 to 3, 13 points).
    2. Case 2: if a < 0 and -b/(2a) lies in [0, 3], then k* = -b/(2a).
    3. Case 3: otherwise the maximum of the fitted curve on [0, 3] is at an endpoint. k* is 0 or 3, whichever has the larger fitted value. If the two are exactly equal, k* = 0. The output reports which case happened and why.
    4. The fine grid is only run in case 2. It is centred on k* rounded to the nearest 0.05, runs from centre - 0.25 to centre + 0.25 in steps of 0.05, is clipped to [0, 3], and uses new games. The fine grid is reported and plotted, but it does not change k*.
    5. If k* = 0, Experiment C uses new games to test the best observed coarse k against k = 0, and the output says so. *My addition:* if the best observed coarse k is also 0, the best observed coarse k above 0 is used.
19. k* is rounded to 3 decimal places before it is used in C.
20. **C:** run for bonus off and on, each with its own tested k. The 2x2 table has rows (adaptive(k) vs break-even, break-even vs break-even) and columns (player A wins, player A loses). The chi-squared test uses no Yates continuity correction, the same as a GDC. Degrees of freedom = 1.
21. **D:** N = 10,000 per k, coarse grid only, bonus off and on. k* for each opponent uses the same decision rule as B. The summary has two rows per opponent:
    - **"k = 0 (break-even rule)":** the observed win rate with its 95% CI. `break_even_edge_over_0.5` = p(0) - 0.5 is the advantage of the break-even rule itself.
    - **"k = k\* (adapting)":** the fitted win rate at k*. `gain_from_adapting` = fitted p(k*) - fitted p(0) is the extra advantage from adapting to rho.

    The observed p for the k* row is only shown when k* is a grid point. The best observed coarse k is also reported.
22. **E:** the main design is mirror matches with N = 20,000. The secondary design is a full matrix with N = 5,000 per pairing. Each unordered pair is played once, because with alternating starts "X vs Y" and "Y vs X" are the same experiment. Game length is measured in turns, rounds and trigger round. The adaptive player in E uses one fixed k = 0.5 (`E_ADAPTIVE_K` in `experiments.py`) for both bonus settings, in the mirror matches and in the full matrix (your decision). This way the "on - off" change for the adaptive row compares the same strategy. k* = 0 could not be used, because adaptive(0) is just break-even again. k = 0.5 is the k tested in Experiment C with the bonus on.
23. **E, rho histogram:** the triggers come from the mirror matches. On a final turn rho is undefined (see 14). So final-turn triggers are shown in a separate panel by their deficit D instead of rho. Normal-turn triggers with rho > 5 are counted in the plot title but not drawn, so the main shape stays readable. All values are in `E_bonus_triggers.csv`.
24. **Step 5 check:** simulated final turns start with own banked score 0 and opponent score D, so the chaser needs S > D. N = 100,000 per D in the full run.
25. The quick run uses N = 1,000 for everything, including the Markov check.

## Summary of key results

These are from the full run (seed 2026). The complete printed output is in `results/full_run_output.txt`.

### Step 1: probability tables

| n | P(farkle) | E(0) | S*_n | Hand value | Difference |
|---|---|---|---|---|---|
| 1 | 0.6667 | 25.00 | 37.50 | 37.5 | 0.00% |
| 2 | 0.4444 | 50.00 | 112.50 | 112.5 | 0.00% |
| 3 | 0.2778 | 86.81 | 312.50 | 312.5 | 0.00% |
| 4 | 0.1574 | 143.60 | 912.25 | 912 | 0.03% |
| 5 | 0.0772 | 226.11 | 2930.42 | 2930 | 0.01% |
| 6 | 0.0231 | 425.19 | 18368.06 | 18368 | 0.00% |

No value differs by more than 1%. For one die with the bonus rule:

| Bonus mode | P(farkle) | E(0) | S*_1 |
|---|---|---|---|
| once | 0.6111 | 29.17 | 47.73 |
| chain | 0.6000 | 30.00 | 50.00 |

### Experiments A and B: finding k (adaptive(k) vs break-even, N = 20,000 per k)

| | Bonus off | Bonus on (chain) |
|---|---|---|
| p at k = 0 (95% CI) | 0.5000 (0.4931 to 0.5070) | 0.5025 (0.4956 to 0.5094) |
| p at k = 3 (95% CI) | 0.4559 (0.4490 to 0.4628) | 0.4597 (0.4528 to 0.4666) |
| Fit p(k) = ak² + bk + c | a = -0.00298, b = -0.00658, c = 0.50213 | a = -0.00104, b = -0.01099, c = 0.50244 |
| R² | 0.984 | 0.983 |
| Vertex -b/(2a) | -1.103 | -5.274 |
| Case | 3 (vertex outside [0, 3]) | 3 (vertex outside [0, 3]) |
| **k\*** | **0** | **0** |
| Fine grid | not run | not run |

The win rate falls steadily as k increases, so adapting to rho does not help against a break-even opponent.

### Experiment C: chi-squared test (new games, N = 20,000 per matchup)

k* = 0 for both settings, so the best observed coarse k was tested against k = 0. With the bonus off this was k = 0.25. With the bonus on, the best observed k was 0 itself, so the best k above 0 was used: k = 0.5.

| | Bonus off | Bonus on |
|---|---|---|
| k tested | 0.25 | 0.5 |
| p, adaptive(k) vs break-even | 0.50005 | 0.49595 |
| p, break-even vs break-even | 0.50030 | 0.49320 |
| χ² (df = 1) | 0.0025 | 0.303 |
| p-value | 0.960 | 0.582 |
| Significant at 5%? | no | no |

### Experiment D: against fixed-threshold opponents (N = 10,000 per k)

"Break-even edge" is p(0) - 0.5, the advantage of the break-even rule itself. "Gain from adapting" is fitted p(k*) - fitted p(0).

| Opponent | Bonus | p at k = 0 (95% CI) | Break-even edge | k* (case) | Gain from adapting |
|---|---|---|---|---|---|
| fixed(T=300) | off | 0.621 (0.611 to 0.630) | +0.121 | 0 (3) | 0 |
| fixed(T=500) | off | 0.694 (0.685 to 0.703) | +0.194 | 0 (3) | 0 |
| fixed(T=1000) | off | 0.758 (0.750 to 0.767) | +0.258 | 0 (3) | 0 |
| fixed(T=2000) | off | 0.930 (0.925 to 0.935) | +0.430 | 0 (3) | 0 |
| fixed(T=300) | on | 0.613 (0.604 to 0.623) | +0.113 | 0.231 (2) | +0.0003 |
| fixed(T=500) | on | 0.661 (0.652 to 0.671) | +0.161 | 0 (3) | 0 |
| fixed(T=1000) | on | 0.721 (0.712 to 0.730) | +0.221 | 0 (3) | 0 |
| fixed(T=2000) | on | 0.903 (0.897 to 0.909) | +0.403 | 0 (3) | 0 |

The break-even rule itself carries almost all of the advantage. Adapting to rho adds nothing measurable.

### Experiment E: game length, mirror matches (N = 20,000 each)

| Strategy | Mean turns, bonus off (SD) | Mean turns, bonus on (SD) | Change, on - off |
|---|---|---|---|
| break-even | 33.68 (7.10) | 33.59 (7.05) | -0.09 |
| fixed(T=300) | 38.09 (6.31) | 38.02 (6.31) | -0.07 |
| fixed(T=500) | 40.32 (8.29) | 39.20 (7.94) | -1.12 |
| fixed(T=1000) | 42.94 (11.34) | 40.85 (10.55) | -2.09 |
| fixed(T=2000) | 69.67 (28.02) | 62.60 (24.96) | -7.08 |
| adaptive (k = 0.5) | 33.79 (6.99) | 33.73 (7.06) | -0.06 |

The 95% CI of each change is about ±0.14 turns for break-even and ±0.5 turns for T = 2000, so the effect is significant for T = 500, 1000 and 2000. The bonus rule shortens games mainly for players who often roll a single die, which is what high fixed thresholds do. Break-even players almost never roll one die: they averaged 0.11 bonus triggers per game, compared with 7.3 per game at T = 2000. The adaptive player (k = 0.5) behaves like break-even here. Its change is -0.06 turns, with a 95% CI of -0.20 to +0.07 (z = -0.91), so it is not significant. It averaged 0.12 bonus triggers per game.

### Step 5: exact final-turn model

| D | P(win), bonus off | P(win), bonus on |
|---|---|---|
| 500 | 0.4799 | 0.4981 |
| 1000 | 0.2830 | 0.3000 |
| 2000 | 0.0855 | 0.0965 |
| 5000 | 0.0031 | 0.0040 |

All 6 checks against 100,000 simulated final turns fall inside the 95% CI. The largest |z| was 1.18.
