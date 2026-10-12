# Farkle IA: handoff summary

This file summarises the simulation project built for my IB Mathematics AI HL Internal Assessment on Farkle: what was built, every decision made, the results, and what is still open. It is written so another chat can pick up the work without the original conversation. The full technical detail (every function, all 25 assumptions) is in `README.md` in the same folder.

Code location: GitHub repo `daniilwerty/test`, branch `claude/sharp-allen-0oby5v`, folder `farkle_ia/`.

## 1. What the project does

A Python simulation of two-player Farkle, using only numpy, scipy, pandas and matplotlib, with a fixed random seed (2026), so every result can be reproduced exactly.

Research questions:
1. Main: how does a bonus-roll house rule change the length of a game for different play styles?
2. Secondary: does an "adaptive" player, who takes more or less risk depending on the score, beat a player who always uses the break-even rule?

Files:
- `scoring.py`: the scoring rules (highest-scoring reading of every roll).
- `probabilities.py`: exact probability tables (Step 1).
- `strategies.py`: break-even, adaptive(k) and fixed-threshold players (Step 2).
- `game.py`: turns, games and logging (Step 3).
- `experiments.py`: Experiments A to E and the Markov check (Step 4 and Step 5 check).
- `markov.py`: exact final-turn model (Step 5).
- `run_all.py`: runs everything. `python run_all.py --quick` is a test run with N = 1,000 (about 20 s). `python run_all.py` is the full run (205 s).
- `tests/test_scoring.py`: 23 hand-checked rolls, plus a test that adaptive(k = 0) makes the same decisions as break-even.
- `results/`: every CSV and PNG from the full run, plus `full_run_output.txt` (the full printed output).

## 2. Game rules used

- 2 players, 6 dice, goal G = 10,000, entry minimum 0.
- After each roll, every scoring die is set aside. Each roll is scored on its own.
- Farkle (no scoring dice): the turn total S is lost.
- Hot dice: if all remaining dice score, S is kept and all 6 dice are rolled again.
- Final round: when a player banks to G or above, the other player gets exactly one final turn and must beat that score strictly. A tie is a loss for the chaser.
- Scoring: single 1 = 100, single 5 = 50, three 1s = 1000, three of a kind (2 to 6) = 100 x face, four of a kind = 1000, five = 2000, six = 3000, straight = 1500, three pairs = 1500, four of a kind + a pair = 1500, two triplets = 2500. Always the highest-scoring reading, so four 1s = 1100 (three 1s + a single 1).
- Bonus rule: when a single remaining die shows a 6, it is not a farkle and the die is rolled again. There are three settings: "off", "once" (a 6 on the bonus roll is a farkle) and "chain" (a 6 on the bonus roll gives another bonus roll). "Bonus on" in the experiments means "chain".

## 3. Strategies

- Break-even: roll if S < S*_n, otherwise bank. S*_n = E(0) / P(farkle) for n dice.
- Adaptive(k): roll if S < S*_n x rho^k, where rho = (G - B) / (G - L). B is the player's own banked score and L is the opponent's. k = 0 is exactly break-even.
- Fixed threshold T: bank as soon as S >= T (T = 300, 500, 1000, 2000), to represent human play styles.
- Rules for every strategy:
  - Final-turn rule: if the opponent has reached G, keep rolling until S > D, where D = opponent's score - own banked score.
  - Always bank as soon as B + S >= G (my decision).
- When the bonus rule is on, players use the bonus-adjusted S*_1 (default setting).

## 4. Decisions I made during the project

1. Bank at G: in a normal turn every strategy banks as soon as B + S >= G.
2. Rounds = turns taken by the starting player = ceil(total turns / 2). The final turn counts. The trigger round (the round in which a player first reaches G) is also recorded.
3. Experiment D: coarse grid only, N = 10,000 per k, bonus off and on, all k* values reported in one table.
4. Experiment E:
   - Main design: mirror matches (both players use the same strategy), N = 20,000 each.
   - Secondary design: a full matrix of every strategy against every other one, N = 5,000 per pairing.
   - Both designs with the bonus rule off and on.
   - The adaptive player uses one k = 0.5 in both bonus settings (mirror matches and the full matrix), so the "on - off" change compares the same strategy. Because k* = 0, adaptive(k*) would just be break-even again. k = 0.5 is the k tested in Experiment C with the bonus on.
5. Decision rule for k* (fixed before the full run):
   1. Fit p(k) = ak^2 + bk + c to the whole coarse grid (k = 0 to 3 in steps of 0.25).
   2. Case 2: if a < 0 and -b/(2a) is in [0, 3], then k* = -b/(2a).
   3. Case 3: otherwise k* is the endpoint (0 or 3) with the larger fitted value. Report which case happened.
   4. Run the fine grid (steps of 0.05) only in case 2, centred on k*.
   5. If k* = 0, Experiment C uses new games to test the best observed coarse k against k = 0.
   6. The Experiment D table includes a k = 0 row for each opponent. This separates the advantage of the break-even rule itself from the advantage of adapting.

## 5. Choices made by Claude that I have not yet confirmed

1. Experiment C, bonus on: the best observed coarse k was 0 itself. So the best k above 0 (k = 0.5) was tested against k = 0.
2. On a final turn rho is undefined (division by zero), so final-turn bonus triggers are shown in a separate histogram panel by their deficit D instead of rho.
3. The chi-squared test uses no Yates correction, the same as a GDC.

## 6. Results (full run, seed 2026)

### Step 1: probability tables

| n | P(farkle) | E(0) | S*_n | Hand value |
|---|---|---|---|---|
| 1 | 0.6667 | 25.00 | 37.50 | 37.5 |
| 2 | 0.4444 | 50.00 | 112.50 | 112.5 |
| 3 | 0.2778 | 86.81 | 312.50 | 312.5 |
| 4 | 0.1574 | 143.60 | 912.25 | 912 |
| 5 | 0.0772 | 226.11 | 2930.42 | 2930 |
| 6 | 0.0231 | 425.19 | 18368.06 | 18368 |

All values are within 1% of my hand calculations. The largest difference is 0.03%. With the bonus rule on one die: "once" gives P(farkle) = 0.6111, E(0) = 29.17 and S*_1 = 47.73; "chain" gives P(farkle) = 0.6000, E(0) = 30.00 and S*_1 = 50.00.

### Experiments A and B: adaptive(k) vs break-even, N = 20,000 per k

- Bonus off: p falls from 0.5000 at k = 0 to 0.4559 at k = 3. The fit is a = -0.00298, b = -0.00658, c = 0.50213, R^2 = 0.984. The vertex is at -1.103, so this is case 3 and k* = 0.
- Bonus on: p falls from 0.5025 at k = 0 to 0.4597 at k = 3. The fit is a = -0.00104, b = -0.01099, c = 0.50244, R^2 = 0.983. The vertex is at -5.274, so this is case 3 and k* = 0.
- The fine grid was not run.
- Conclusion: adapting to rho does not help against a break-even player. Higher k is steadily worse.

### Experiment C: chi-squared test on new games, N = 20,000 per matchup

- Bonus off: k = 0.25 vs k = 0. p(adaptive) = 0.50005 and p(break-even mirror) = 0.50030. chi^2 = 0.0025, df = 1, p-value = 0.960. Not significant.
- Bonus on: k = 0.5 vs k = 0. p(adaptive) = 0.49595 and p(break-even mirror) = 0.49320. chi^2 = 0.303, df = 1, p-value = 0.582. Not significant.

### Experiment D: against fixed-threshold players, N = 10,000 per k

| Opponent | Bonus | p at k = 0 | Break-even edge (p - 0.5) | k* (case) | Gain from adapting |
|---|---|---|---|---|---|
| T = 300 | off | 0.621 | +0.121 | 0 (3) | 0 |
| T = 500 | off | 0.694 | +0.194 | 0 (3) | 0 |
| T = 1000 | off | 0.758 | +0.258 | 0 (3) | 0 |
| T = 2000 | off | 0.930 | +0.430 | 0 (3) | 0 |
| T = 300 | on | 0.613 | +0.113 | 0.231 (2) | +0.0003 |
| T = 500 | on | 0.661 | +0.161 | 0 (3) | 0 |
| T = 1000 | on | 0.721 | +0.221 | 0 (3) | 0 |
| T = 2000 | on | 0.903 | +0.403 | 0 (3) | 0 |

The break-even rule itself carries the advantage. Adapting adds nothing measurable.

### Experiment E: game length, mirror matches, N = 20,000 each (main research question)

| Strategy | Mean turns, bonus off (SD) | Mean turns, bonus on (SD) | Change |
|---|---|---|---|
| break-even | 33.68 (7.10) | 33.59 (7.05) | -0.09 (not significant) |
| fixed T = 300 | 38.09 (6.31) | 38.02 (6.31) | -0.07 (not significant) |
| fixed T = 500 | 40.32 (8.29) | 39.20 (7.94) | -1.12 |
| fixed T = 1000 | 42.94 (11.34) | 40.85 (10.55) | -2.09 |
| fixed T = 2000 | 69.67 (28.02) | 62.60 (24.96) | -7.08 |
| adaptive (k = 0.5) | 33.79 (6.99) | 33.73 (7.06) | -0.06 (not significant) |

- The bonus rule shortens games mainly for high fixed thresholds, because those players often roll a single die. Bonus triggers per game: break-even 0.11, T = 300 0.14, T = 500 2.10, T = 1000 3.43, T = 2000 7.34.
- For the adaptive row the 95% CI of the change is -0.20 to +0.07 turns (z = -0.91), so the bonus rule has no significant effect on its game length, the same as for break-even. It triggered the bonus rule 0.12 times per game.
- Rounds follow the same pattern (rounds = about half the turns).
- The full strategy-vs-strategy matrices are in `results/E_matrix_mean_turns_bonus_off.csv` and `results/E_matrix_mean_turns_bonus_on.csv`.

### Step 5: exact final-turn model (Markov chain)

P(chaser wins the final turn):

| D | Bonus off | Bonus on |
|---|---|---|
| 500 | 0.4799 | 0.4981 |
| 1000 | 0.2830 | 0.3000 |
| 2000 | 0.0855 | 0.0965 |
| 5000 | 0.0031 | 0.0040 |

All 6 checks against 100,000 simulated final turns fall inside the 95% CI. The largest |z| was 1.18.

## 7. Plots (in `results/`)

- `B_win_rate_vs_k_bonus_off.png`, `B_win_rate_vs_k_bonus_on.png`: win rate against k with 95% error bars and the fitted quadratic.
- `D_win_rate_vs_k_fixed_opponents.png`: win rate against k for each fixed-threshold opponent.
- `E_game_length_by_strategy.png`: mean game length by strategy (bonus off and on), and the change on - off.
- `E_rho_histogram_bonus_triggers.png`: rho at each bonus trigger. Final-turn triggers are shown separately by deficit D.
- `markov_final_turn_curve.png`: exact P(win final turn) against D, for bonus off and on.

## 8. Next steps

1. Write the IA. A possible order: Farkle rules, then the probability tables and break-even derivation, then strategies, then the simulation method, then the results of A to E, then the Markov model and its check, then conclusion and limitations.
2. Limitations worth discussing:
   - S*_n only looks one roll ahead.
   - Strategies only depend on S, n, B and L.
   - Players always set aside every scoring die.
   - The k = 0 result may depend on the shape of the threshold formula S*_n x rho^k.
