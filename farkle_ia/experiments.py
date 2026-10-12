"""
experiments.py - Step 4 and the Step 5 checks: all the experiments.

Each experiment function saves its results as CSV files and its plots as
PNG files in an output folder, and returns the main numbers so run_all.py
can print a summary.

    experiment_A  find the best k (coarse grid, then a finer grid)
    experiment_B  fit a quadratic p(k) = a k^2 + b k + c near the peak
    experiment_C  chi-squared test on brand new games
    experiment_D  robustness against fixed-threshold opponents
    experiment_E  game length (mirror matches and a full matrix)
    experiment_markov  exact final-turn curves and simulation check
"""

import math
import os

import matplotlib
matplotlib.use("Agg")            # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from game import DiceRoller, GameSettings, make_rng, play_game, play_turn
from markov import win_probability
from probabilities import bonus_table, probability_table
from strategies import Adaptive, BreakEven, FixedThreshold

# The two bonus settings that every experiment is run with.
# "on" means the default bonus mode, "chain".
BONUS_SETTINGS = {"off": "off", "on": "chain"}

COARSE_K_GRID = [round(0.25 * i, 2) for i in range(13)]   # 0, 0.25, ..., 3
FIXED_T_VALUES = [300, 500, 1000, 2000]
Z_95 = 1.959964   # z value for a 95% confidence interval


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def save_csv(table, folder, filename):
    os.makedirs(folder, exist_ok=True)
    table.to_csv(os.path.join(folder, filename), index=False)


def save_plot(figure, folder, filename):
    os.makedirs(folder, exist_ok=True)
    figure.savefig(os.path.join(folder, filename), dpi=150, bbox_inches="tight")
    plt.close(figure)


def make_settings(bonus_label):
    """Game settings for bonus 'off' or 'on' (on = chain)."""
    return GameSettings(bonus_mode=BONUS_SETTINGS[bonus_label])


def make_strategy(kind, value, settings):
    """
    Build a strategy from a short description:
        ("break-even", None), ("adaptive", k) or ("fixed", T).
    The settings tell the strategy which S*_n values to use.
    """
    if kind == "break-even":
        return BreakEven(settings.bonus_mode, settings.bonus_aware)
    if kind == "adaptive":
        return Adaptive(value, settings.bonus_mode, settings.bonus_aware)
    if kind == "fixed":
        return FixedThreshold(value)
    raise ValueError(kind)


def play_match(strategy_a, strategy_b, n_games, settings, label):
    """
    Play n_games between player A and player B, alternating who starts
    (A starts the even-numbered games, B the odd-numbered ones).

    Returns three tables: one row per game, one row per bonus-rule trigger,
    and one row per very large rho value.
    """
    roller = DiceRoller(make_rng(label))
    bonus_log = []
    large_rho_log = []
    games = []
    for game_id in range(n_games):
        a_starts = (game_id % 2 == 0)
        games.append(play_game(strategy_a, strategy_b, a_starts, settings, roller,
                               bonus_log, large_rho_log, game_id))
    return pd.DataFrame(games), pd.DataFrame(bonus_log), pd.DataFrame(large_rho_log)


def win_rate_summary(games):
    """
    Player A's win rate p with its standard error and 95% confidence interval.
        SE = sqrt(p (1 - p) / N)
        CI = p +/- 1.96 x SE
    """
    n = len(games)
    wins = int((games["winner"] == "A").sum())
    p = wins / n
    se = math.sqrt(p * (1 - p) / n)
    return {"wins": wins, "N": n, "p": p, "SE": se,
            "CI_low": p - Z_95 * se, "CI_high": p + Z_95 * se}


def k_sweep(k_values, opponent_kind, opponent_value, n_games, bonus_label, tag):
    """
    Play adaptive(k) (player A) against one opponent (player B) for every k.
    Returns one row per k with the win-rate summary.
    """
    settings = make_settings(bonus_label)
    opponent = make_strategy(opponent_kind, opponent_value, settings)
    rows = []
    for k in k_values:
        player = make_strategy("adaptive", k, settings)
        label = f"{tag} bonus={bonus_label} opponent={opponent.name} k={k:g}"
        games, bonus_log, large_rho_log = play_match(player, opponent, n_games,
                                                     settings, label)
        row = {"k": k, "bonus": bonus_label, "opponent": opponent.name}
        row.update(win_rate_summary(games))
        row["mean_turns"] = games["turns"].mean()
        row["bonus_triggers"] = len(bonus_log)
        row["large_rho_turns"] = len(large_rho_log)
        row["max_rho"] = large_rho_log["rho"].max() if len(large_rho_log) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fit_quadratic(k_values, win_rates):
    """
    Least-squares fit of p(k) = a k^2 + b k + c.

    Returns a, b, c, R^2 and k* = -b / (2a), the k at the top of the parabola.
    If a >= 0 the parabola has no maximum, so k* is set to NaN.

    R^2 = 1 - (sum of squared residuals) / (total sum of squares).
    """
    k = np.asarray(k_values, dtype=float)
    p = np.asarray(win_rates, dtype=float)
    a, b, c = np.polyfit(k, p, deg=2)          # least squares
    predicted = a * k ** 2 + b * k + c
    ss_residual = float(((p - predicted) ** 2).sum())
    ss_total = float(((p - p.mean()) ** 2).sum())
    r_squared = 1 - ss_residual / ss_total if ss_total > 0 else np.nan
    k_star = -b / (2 * a) if a < 0 else np.nan
    return {"a": a, "b": b, "c": c, "R_squared": r_squared, "k_star": k_star}


# ---------------------------------------------------------------------------
# Step 1: probability tables
# ---------------------------------------------------------------------------

def step1_probabilities(folder):
    table = probability_table()
    bonus = bonus_table()
    save_csv(table, folder, "step1_probability_table.csv")
    save_csv(bonus, folder, "step1_one_die_bonus.csv")
    return table, bonus


# ---------------------------------------------------------------------------
# Experiment A: finding k
# ---------------------------------------------------------------------------

def experiment_A(folder, n_games):
    """
    Adaptive(k) vs break-even for k = 0, 0.25, ..., 3, then a finer grid
    (steps of 0.05) from best k - 0.25 to best k + 0.25. Done with the bonus
    rule off and on. The fine grid uses new games.
    """
    coarse_tables = []
    fine_tables = []
    for bonus_label in BONUS_SETTINGS:
        coarse = k_sweep(COARSE_K_GRID, "break-even", None, n_games, bonus_label,
                         "A coarse")
        best_k = coarse.loc[coarse["p"].idxmax(), "k"]
        # Steps of 0.05 from best_k - 0.25 to best_k + 0.25, never below 0.
        fine_grid = [round(best_k + 0.05 * step, 2) for step in range(-5, 6)
                     if best_k + 0.05 * step >= -1e-9]
        fine_grid = [abs(k) for k in fine_grid]   # turn -0.0 into 0.0
        fine = k_sweep(fine_grid, "break-even", None, n_games, bonus_label, "A fine")
        fine["best_coarse_k"] = best_k
        coarse_tables.append(coarse)
        fine_tables.append(fine)

    coarse_all = pd.concat(coarse_tables, ignore_index=True)
    fine_all = pd.concat(fine_tables, ignore_index=True)
    save_csv(coarse_all, folder, "A_coarse_grid.csv")
    save_csv(fine_all, folder, "A_fine_grid.csv")
    return coarse_all, fine_all


# ---------------------------------------------------------------------------
# Experiment B: quadratic fit near the peak
# ---------------------------------------------------------------------------

def experiment_B(folder, coarse_all, fine_all):
    """
    Fit p(k) = a k^2 + b k + c to the fine-grid points (the points near the
    peak) and plot win rate against k with 95% error bars and the fitted curve.
    """
    fits = []
    for bonus_label in BONUS_SETTINGS:
        fine = fine_all[fine_all["bonus"] == bonus_label]
        coarse = coarse_all[coarse_all["bonus"] == bonus_label]
        fit = fit_quadratic(fine["k"], fine["p"])
        fit["bonus"] = bonus_label
        fit["points_used"] = len(fine)
        fit["k_range"] = f"{fine['k'].min():g} to {fine['k'].max():g}"
        if math.isnan(fit["k_star"]):
            # No maximum in the fitted curve: fall back to the best observed k.
            fit["k_star_used"] = float(fine.loc[fine["p"].idxmax(), "k"])
            fit["note"] = "a >= 0, no maximum; k* = best observed k"
        else:
            fit["k_star_used"] = fit["k_star"]
            fit["note"] = ""
        fits.append(fit)

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.errorbar(coarse["k"], coarse["p"], yerr=Z_95 * coarse["SE"], fmt="o",
                      color="grey", alpha=0.6, capsize=3, label="coarse grid (95% CI)")
        axis.errorbar(fine["k"], fine["p"], yerr=Z_95 * fine["SE"], fmt="o",
                      color="tab:blue", capsize=3, label="fine grid (95% CI)")
        k_line = np.linspace(fine["k"].min(), fine["k"].max(), 200)
        axis.plot(k_line, fit["a"] * k_line ** 2 + fit["b"] * k_line + fit["c"],
                  color="tab:red",
                  label=f"fit: {fit['a']:.4f}k² {fit['b']:+.4f}k {fit['c']:+.4f}"
                        f"  (R² = {fit['R_squared']:.3f})")
        axis.axvline(fit["k_star_used"], color="tab:red", linestyle=":",
                     label=f"k* = {fit['k_star_used']:.3f}")
        axis.axhline(0.5, color="black", linewidth=0.8)
        axis.set_xlabel("k")
        axis.set_ylabel("win rate of adaptive(k) vs break-even")
        axis.set_title(f"Win rate against k (bonus rule {bonus_label})")
        axis.legend(fontsize=8)
        save_plot(figure, folder, f"B_win_rate_vs_k_bonus_{bonus_label}.png")

    fits = pd.DataFrame(fits)
    save_csv(fits, folder, "B_quadratic_fit.csv")
    return fits


# ---------------------------------------------------------------------------
# Experiment C: chi-squared test of independence on new games
# ---------------------------------------------------------------------------

def experiment_C(folder, n_games, k_star_by_bonus):
    """
    New games: adaptive(k*) vs break-even, and break-even vs break-even.
    2x2 table:            player A wins   player A loses
        adaptive vs BE         ...             ...
        BE vs BE               ...             ...
    Chi-squared test of independence (no Yates continuity correction, the
    same as a GDC). Degrees of freedom = (2 - 1)(2 - 1) = 1.
    """
    rows = []
    tables = []
    for bonus_label in BONUS_SETTINGS:
        settings = make_settings(bonus_label)
        k_star = k_star_by_bonus[bonus_label]
        adaptive = make_strategy("adaptive", k_star, settings)
        break_even = make_strategy("break-even", None, settings)

        games_1, _, _ = play_match(adaptive, break_even, n_games, settings,
                                   f"C bonus={bonus_label} adaptive vs break-even")
        games_2, _, _ = play_match(break_even, break_even, n_games, settings,
                                   f"C bonus={bonus_label} break-even vs break-even")

        wins_1 = int((games_1["winner"] == "A").sum())
        wins_2 = int((games_2["winner"] == "A").sum())
        observed = np.array([[wins_1, n_games - wins_1],
                             [wins_2, n_games - wins_2]])
        chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)

        tables.append(pd.DataFrame({
            "bonus": bonus_label,
            "matchup": [f"{adaptive.name} vs break-even", "break-even vs break-even"],
            "A_wins": observed[:, 0], "A_losses": observed[:, 1],
            "expected_A_wins": expected[:, 0], "expected_A_losses": expected[:, 1],
        }))
        rows.append({"bonus": bonus_label, "k_star": k_star,
                     "p_adaptive": wins_1 / n_games, "p_break_even_mirror": wins_2 / n_games,
                     "chi2": chi2, "dof": dof, "p_value": p_value,
                     "significant_at_5_percent": p_value < 0.05})

    results = pd.DataFrame(rows)
    save_csv(pd.concat(tables, ignore_index=True), folder, "C_contingency_tables.csv")
    save_csv(results, folder, "C_chi_squared.csv")
    return results


# ---------------------------------------------------------------------------
# Experiment D: robustness against fixed-threshold opponents
# ---------------------------------------------------------------------------

def experiment_D(folder, n_games):
    """
    Coarse k grid against fixed(T) for T = 300, 500, 1000, 2000, bonus off
    and on. k* for each opponent comes from a quadratic fit to the best
    coarse k and its neighbours (5 points: best k +/- 0.5); the best observed
    k is reported as well.
    """
    sweeps = []
    summary = []
    for bonus_label in BONUS_SETTINGS:
        for T in FIXED_T_VALUES:
            sweep = k_sweep(COARSE_K_GRID, "fixed", T, n_games, bonus_label, "D")
            sweeps.append(sweep)

            best_index = sweep["p"].idxmax()
            best_k = sweep.loc[best_index, "k"]
            near = sweep[(sweep["k"] >= best_k - 0.5 - 1e-9) &
                         (sweep["k"] <= best_k + 0.5 + 1e-9)]
            fit = fit_quadratic(near["k"], near["p"])
            summary.append({
                "bonus": bonus_label, "opponent": f"fixed(T={T})",
                "best_observed_k": best_k, "best_observed_p": sweep.loc[best_index, "p"],
                "k_star_fit": fit["k_star"], "fit_R_squared": fit["R_squared"],
                "p_at_k0": sweep.loc[sweep["k"] == 0, "p"].iloc[0],
            })

    sweeps = pd.concat(sweeps, ignore_index=True)
    summary = pd.DataFrame(summary)
    save_csv(sweeps, folder, "D_grid_vs_fixed.csv")
    save_csv(summary, folder, "D_k_star_summary.csv")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, bonus_label in zip(axes, BONUS_SETTINGS):
        for T in FIXED_T_VALUES:
            part = sweeps[(sweeps["bonus"] == bonus_label) &
                          (sweeps["opponent"] == f"fixed(T={T})")]
            axis.errorbar(part["k"], part["p"], yerr=Z_95 * part["SE"], marker="o",
                          capsize=2, label=f"vs fixed T = {T}")
        axis.set_title(f"Bonus rule {bonus_label}")
        axis.set_xlabel("k")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("win rate of adaptive(k)")
    save_plot(figure, folder, "D_win_rate_vs_k_fixed_opponents.png")
    return summary


# ---------------------------------------------------------------------------
# Experiment E: game length
# ---------------------------------------------------------------------------

def strategy_list(k_star):
    """The six strategies compared in Experiment E."""
    return ([("break-even", None)] + [("fixed", T) for T in FIXED_T_VALUES]
            + [("adaptive", k_star)])


def length_summary(games):
    return {"N": len(games),
            "mean_turns": games["turns"].mean(), "sd_turns": games["turns"].std(),
            "mean_rounds": games["rounds"].mean(), "sd_rounds": games["rounds"].std(),
            "mean_trigger_round": games["trigger_round"].mean()}


def experiment_E(folder, n_mirror, n_matrix, k_star_by_bonus):
    """
    Main design: mirror matches (both players use the same strategy).
    Secondary design: every strategy against every other one.
    Both with the bonus rule off and on. Also collects every bonus-rule
    trigger from the mirror matches for the rho histogram.
    """
    mirror_rows = []
    matrix_rows = []
    all_triggers = []
    all_large_rho = []

    for bonus_label in BONUS_SETTINGS:
        settings = make_settings(bonus_label)
        specs = strategy_list(k_star_by_bonus[bonus_label])
        players = [make_strategy(kind, value, settings) for kind, value in specs]

        # Main design: mirror matches.
        for player in players:
            label = f"E mirror bonus={bonus_label} {player.name}"
            games, triggers, large_rho = play_match(player, player, n_mirror,
                                                    settings, label)
            row = {"bonus": bonus_label, "strategy": player.name}
            row.update(length_summary(games))
            row["bonus_triggers_per_game"] = len(triggers) / len(games)
            mirror_rows.append(row)
            if len(triggers):
                triggers["bonus"] = bonus_label
                triggers["match"] = player.name
                all_triggers.append(triggers)
            if len(large_rho):
                large_rho["bonus"] = bonus_label
                all_large_rho.append(large_rho)

        # Secondary design: every pairing (each unordered pair once; with
        # alternating starts, X vs Y and Y vs X are the same experiment).
        for i, player_1 in enumerate(players):
            for player_2 in players[i:]:
                label = f"E matrix bonus={bonus_label} {player_1.name} vs {player_2.name}"
                games, _, _ = play_match(player_1, player_2, n_matrix, settings, label)
                row = {"bonus": bonus_label, "strategy_A": player_1.name,
                       "strategy_B": player_2.name}
                row.update(length_summary(games))
                row["win_rate_A"] = (games["winner"] == "A").mean()
                matrix_rows.append(row)

    mirror = pd.DataFrame(mirror_rows)
    matrix = pd.DataFrame(matrix_rows)
    triggers = pd.concat(all_triggers, ignore_index=True) if all_triggers else pd.DataFrame()
    large_rho = pd.concat(all_large_rho, ignore_index=True) if all_large_rho else pd.DataFrame()

    save_csv(mirror, folder, "E_mirror_game_length.csv")
    save_csv(matrix, folder, "E_matrix_game_length_long.csv")
    for bonus_label in BONUS_SETTINGS:
        part = matrix[matrix["bonus"] == bonus_label]
        # Fill both halves of the square table (the matrix is symmetric).
        flipped = part.rename(columns={"strategy_A": "strategy_B", "strategy_B": "strategy_A"})
        square = (pd.concat([part, flipped]).drop_duplicates(["strategy_A", "strategy_B"])
                  .pivot(index="strategy_A", columns="strategy_B", values="mean_turns"))
        order = list(mirror[mirror["bonus"] == bonus_label]["strategy"])
        square = square.loc[order, order].reset_index()
        save_csv(square, folder, f"E_matrix_mean_turns_bonus_{bonus_label}.csv")
    save_csv(triggers, folder, "E_bonus_triggers.csv")
    save_csv(large_rho, folder, "E_large_rho_log.csv")

    _plot_game_length(folder, mirror)
    _plot_rho_histogram(folder, triggers)
    return mirror, matrix, triggers


def _plot_game_length(folder, mirror):
    """Mean game length per strategy (off vs on), and the change on - off."""
    off = mirror[mirror["bonus"] == "off"].reset_index(drop=True)
    on = mirror[mirror["bonus"] == "on"].reset_index(drop=True)
    # Names line up because both lists are built in the same order.
    names = [name if not name.startswith("adaptive") else "adaptive(k*)"
             for name in off["strategy"]]
    x = np.arange(len(names))

    se_off = off["sd_turns"] / np.sqrt(off["N"])
    se_on = on["sd_turns"] / np.sqrt(on["N"])
    change = on["mean_turns"] - off["mean_turns"]
    se_change = np.sqrt(se_off ** 2 + se_on ** 2)

    figure, (axis_1, axis_2) = plt.subplots(1, 2, figsize=(12, 4.5))
    axis_1.bar(x - 0.2, off["mean_turns"], 0.4, yerr=Z_95 * se_off, capsize=3,
               label="bonus off")
    axis_1.bar(x + 0.2, on["mean_turns"], 0.4, yerr=Z_95 * se_on, capsize=3,
               label="bonus on (chain)")
    axis_1.set_xticks(x, names, rotation=30, ha="right")
    axis_1.set_ylabel("mean game length (turns)")
    axis_1.set_title("Mean game length, mirror matches (95% CI)")
    axis_1.legend()

    axis_2.bar(x, change, yerr=Z_95 * se_change, capsize=3, color="tab:green")
    axis_2.axhline(0, color="black", linewidth=0.8)
    axis_2.set_xticks(x, names, rotation=30, ha="right")
    axis_2.set_ylabel("change in mean length (turns), on - off")
    axis_2.set_title("Effect of the bonus rule on game length (95% CI)")
    save_plot(figure, folder, "E_game_length_by_strategy.png")


def _plot_rho_histogram(folder, triggers):
    """
    Histogram of rho at each bonus-rule trigger.

    On a final turn the opponent has already reached G, so rho = (G-B)/(G-L)
    has a zero or negative denominator and is undefined (NaN). Final-turn
    triggers are therefore shown in a second panel, using the deficit D
    (points the chaser needs) instead of rho.
    """
    if triggers.empty:
        return
    normal = triggers[~triggers["final_turn"]]
    final = triggers[triggers["final_turn"]]

    figure, (axis_1, axis_2) = plt.subplots(1, 2, figsize=(12, 4.5))
    rho = normal["rho"].dropna()
    rho_shown = rho[rho <= 5]
    axis_1.hist(rho_shown, bins=50, color="tab:blue")
    axis_1.set_xlabel("rho at trigger")
    axis_1.set_ylabel("number of triggers")
    axis_1.set_title(f"Normal turns: {len(rho)} triggers "
                     f"({len(rho) - len(rho_shown)} with rho > 5 not shown)")

    axis_2.hist(final["deficit"], bins=40, color="tab:orange")
    axis_2.set_xlabel("deficit D at trigger (rho undefined on final turns)")
    axis_2.set_title(f"Final turns: {len(final)} triggers")
    save_plot(figure, folder, "E_rho_histogram_bonus_triggers.png")


# ---------------------------------------------------------------------------
# Step 5: exact final-turn model and its check
# ---------------------------------------------------------------------------

def experiment_markov(folder, n_validation):
    """
    Exact P(win final turn) for D = 50 to 5000 (step 50), bonus off and on,
    then compare with simulated final turns at D = 500, 1000 and 2000.
    """
    deficits = list(range(50, 5001, 50))
    curve = pd.DataFrame({"D": deficits})
    for bonus_label, mode in BONUS_SETTINGS.items():
        # Going up in D means each value reuses the smaller ones already stored.
        curve[f"P_win_bonus_{bonus_label}"] = [win_probability(D, 6, mode) for D in deficits]
    save_csv(curve, folder, "markov_final_turn_curve.csv")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for bonus_label in BONUS_SETTINGS:
        axis.plot(curve["D"], curve[f"P_win_bonus_{bonus_label}"],
                  label=f"bonus {bonus_label}")
    axis.set_xlabel("deficit D (points the chaser must beat)")
    axis.set_ylabel("P(chaser wins final turn)")
    axis.set_title("Exact final-turn winning probability")
    axis.legend()
    save_plot(figure, folder, "markov_final_turn_curve.png")

    rows = []
    for bonus_label in BONUS_SETTINGS:
        settings = make_settings(bonus_label)
        chaser = BreakEven(settings.bonus_mode)   # the strategy does not matter on a final turn
        for D in (500, 1000, 2000):
            roller = DiceRoller(make_rng(f"markov check bonus={bonus_label} D={D}"))
            wins = 0
            for _ in range(n_validation):
                # Own banked score 0, opponent on D: the chaser needs S > D.
                if play_turn(chaser, 0, D, True, settings, roller) > 0:
                    wins += 1
            p_sim = wins / n_validation
            se = math.sqrt(p_sim * (1 - p_sim) / n_validation)
            exact = win_probability(D, 6, settings.bonus_mode)
            z = (p_sim - exact) / se if se > 0 else np.nan
            rows.append({"bonus": bonus_label, "D": D, "exact": exact,
                         "simulated": p_sim, "N": n_validation, "SE": se,
                         "difference": p_sim - exact, "z_score": z,
                         "within_95_CI": abs(z) < Z_95})
    check = pd.DataFrame(rows)
    save_csv(check, folder, "markov_validation.csv")
    return curve, check
